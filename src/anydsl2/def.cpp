#include "anydsl2/def.h"

#include <algorithm>

#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/primop.h"
#include "anydsl2/printer.h"
#include "anydsl2/type.h"
#include "anydsl2/world.h"
#include "anydsl2/util/for_all.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

Def::~Def() { 
    for (size_t i = 0, e = ops().size(); i != e; ++i)
        unregister_use(i);

    for_all (use, uses_) {
        size_t i = use.index();
        assert(use.def()->ops()[i] == this && "use points to incorrect def");
        const_cast<Def*>(use.def())->set(i, 0);
    }
}

void Def::set_op(size_t i, const Def* def) {
    assert(!op(i) && "already set");
    assert(std::find(def->uses_.begin(), def->uses_.end(), Use(i, this)) == def->uses_.end() && "already in use set");
    def->uses_.push_back(Use(i, this));
    set(i, def);
}

void Def::unset_op(size_t i) {
    assert(op(i) && "must be set");
    unregister_use(i);
    set(i, 0);
}

void Def::unregister_use(size_t i) const {
    if (const Def* def = op(i)) {
        assert(std::find(def->uses_.begin(), def->uses_.end(), Use(i, this)) != def->uses_.end() && "must be in use set");
        def->uses_.erase(std::find(def->uses_.begin(), def->uses_.end(), Use(i, this)));
    }
}

bool Def::is_const() const {
    if (node_kind() == Node_Param)
        return false;

    if (empty() || node_kind() == Node_Lambda)
        return true;

    for (size_t i = 0, e = size(); i != e; ++i)
        if (!op(i)->is_const())
            return false;

    return true;
}

void Def::update(size_t i, const Def* def) {
    unset_op(i);
    set_op(i, def);
    //Uses::iterator it = def->uses_.erase(std::find(def->uses_.begin(), def->uses_.end(), Use(this, i)));
    //*it = Use(def, i);
    //set(i, def);
}

void Def::update(ArrayRef<const Def*> defs) {
    assert(size() == defs.size() && "sizes do not match");

    for (size_t i = 0, e = size(); i != e; ++i)
        update(i, defs[i]);
}

Array<Use> Def::copy_uses() const {
    Array<Use> result(uses().size());
    std::copy(uses().begin(), uses().end(), result.begin());
    return result;
}

bool Def::is_primlit(int val) const {
    if (const PrimLit* lit = this->isa<PrimLit>()) {
        Box box = lit->box();
        switch (lit->primtype_kind()) {
#define ANYDSL2_UF_TYPE(T) case PrimType_##T: return box.get_##T() == T(val);
#include "anydsl2/tables/primtypetable.h"
        }
    }
    return false;
}

World& Def::world() const { return type_->world(); }
const Def* Def::op_via_lit(const Def* def) const { return op(def->primlit_value<size_t>()); }
Lambda* Def::as_lambda() const { return const_cast<Lambda*>(scast<Lambda>(this)); }
Lambda* Def::isa_lambda() const { return const_cast<Lambda*>(dcast<Lambda>(this)); }
int Def::order() const { return type()->order(); }
void Def::replace(const Def* with) const { world().replace(this, with); }

size_t hash_node(const DefTuple0& tuple) { return hash_kind_type_size(tuple, 0); }
size_t hash_node(const DefTuple1& tuple) {
    size_t seed = hash_kind_type_size(tuple, 1);
    boost::hash_combine(seed, tuple.get<2>());
    return seed;
}
size_t hash_node(const DefTuple2& tuple) {
    size_t seed = hash_kind_type_size(tuple, 2);
    boost::hash_combine(seed, tuple.get<2>());
    boost::hash_combine(seed, tuple.get<3>());
    return seed;
}
size_t hash_node(const DefTuple3& tuple) {
    size_t seed = hash_kind_type_size(tuple, 3);
    boost::hash_combine(seed, tuple.get<2>());
    boost::hash_combine(seed, tuple.get<3>());
    boost::hash_combine(seed, tuple.get<4>());
    return seed;
}
size_t hash_node(const DefTupleN& tuple) {
    ArrayRef<const Def*> ops = tuple.get<2>();
    size_t seed = hash_kind_type_size(tuple, ops.size());
    for_all (op, ops)
        boost::hash_combine(seed, op);
    return seed;
}

bool equal_node(const DefTuple0& tuple, const Node* node) {
    const Def* def = node->as<Def>();
    return equal_kind_type_size(tuple, 0, def);
}
bool equal_node(const DefTuple1& tuple, const Node* node) {
    const Def* def = node->as<Def>();
    return equal_kind_type_size(tuple, 1, def) && tuple.get<2>() == def->op(0);
}
bool equal_node(const DefTuple2& tuple, const Node* node) {
    const Def* def = node->as<Def>();
    return equal_kind_type_size(tuple, 2, def) 
        && tuple.get<2>() == def->op(0) && tuple.get<3>() == def->op(1);
}
bool equal_node(const DefTuple3& tuple, const Node* node) {
    const Def* def = node->as<Def>();
    return equal_kind_type_size(tuple, 3, def) 
        && tuple.get<2>() == def->op(0) && tuple.get<3>() == def->op(1) && tuple.get<4>() == def->op(2);
}
bool equal_node(const DefTupleN& tuple, const Node* node) {
    const Def* def = node->as<Def>();
    return equal_kind_type_size(tuple, tuple.get<2>().size(), def) && tuple.get<2>() == def->ops();
}

std::ostream& operator << (std::ostream& o, const anydsl2::Def* def) {
    Printer p(o, false);
    def->vdump(p);
    return p.o;
}

//------------------------------------------------------------------------------

Param::Param(const Type* type, Lambda* lambda, size_t index, const std::string& name)
    : Def(Node_Param, 0, type, name)
    , lambda_(lambda)
    , index_(index)
{}

Peeks Param::peek() const {
    size_t x = index();
    Lambda* l = lambda();
    LambdaSet preds = l->direct_preds();
    Peeks result(preds.size());
    for_all2 (&res, result, pred, preds)
        res = Peek(pred->arg(x), pred);

    return result;
}

bool Param::equal(const Node* other) const {
    return Def::equal(other) 
        && index()  == other->as<Param>()->index() 
        && lambda() == other->as<Param>()->lambda();
}

size_t Param::hash() const {
    size_t seed = Def::hash();
    boost::hash_combine(seed, index());
    boost::hash_combine(seed, lambda());
    return seed;
}

//------------------------------------------------------------------------------

} // namespace anydsl2
