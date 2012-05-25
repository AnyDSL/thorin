#include "anydsl/air/def.h"

#include <typeinfo>

#include "anydsl/air/primop.h"
#include "anydsl/air/type.h"
#include "anydsl/air/use.h"
#include "anydsl/util/foreach.h"

namespace anydsl {


//------------------------------------------------------------------------------

Use::Use(AIRNode* parent, Def* def)
    : AIRNode(Index_Use)
    , def_(def) 
    , parent_(parent)
{
    def_->registerUse(this);
}

Use::~Use() {
    def_->unregisterUse(this);
}

World& Use::world() {
    return def_->world();
}

//------------------------------------------------------------------------------

Ops::Ops()
    : sentinel_(new Node())
    , size_(0)
{
    sentinel_->next_ = sentinel_;
    sentinel_->prev_ = sentinel_;
}

Ops::~Ops() {
    clear();
    anydsl_assert(sentinel_->isSentinel_, "this must be the sentinel");
    delete sentinel_;
}

Ops::iterator Ops::insert(Ops::iterator pos, Def* parent, Def* def) {
    Node* newNode = new UseNode(parent, def);
    Node* n = pos.n_;

    newNode->next_ = n;
    newNode->prev_ = n->prev_;

    n->prev_ = newNode;
    newNode->prev_->next_ = newNode;

    ++size_;

    return iterator(newNode);
}

Ops::iterator Ops::erase(Ops::iterator pos) {
    UseNode* n = (UseNode*) pos.n_;
    anydsl_assert(!n->isSentinel_, "this must not be the sentinel");

    iterator res(n->next_);
    n->next_->prev_ = n->prev_;
    n->prev_->next_ = n->next_;
    delete n;

    --size_;

    return res;
}

void Ops::clear() {
    Node* i = head();

    while (i != sentinel_) {
        anydsl_assert(!i->isSentinel_, "this must not be the sentinel");
        UseNode* cur = (UseNode*) i;
        i = i->next_;
        delete cur;
    }

    size_ = 0;
    sentinel_->next_ = sentinel_;
    sentinel_->prev_ = sentinel_;
}

//------------------------------------------------------------------------------

void Def::registerUse(Use* use) { 
    anydsl_assert(use->def() == this, "use does not point to this def");
    anydsl_assert(uses_.find(use) == uses_.end(), "must not be inside the use list");
    uses_.insert(use);
}

void Def::unregisterUse(Use* use) { 
    anydsl_assert(use->def() == this, "use does not point to this def");
    anydsl_assert(uses_.find(use) != uses_.end(), "must be inside the use list");
    uses_.erase(use);
}

World& Def::world() const { 
    return type_->world(); 
}

bool ValueNumber::operator == (const ValueNumber& vn) const {
    if (index != vn.index)
        return false;

    if (hasMore(index)) {
        if (size != vn.size)
            return false;

        bool result = true;
        for (size_t i = 0, e = size; i != e && result; ++i)
            result &= more[i] == vn.more[i];

        return result;
    }

    return op1 == vn.op1 && op2 == vn.op2 && op3 == vn.op3;
}

size_t hash_value(const ValueNumber& vn) {
    size_t seed = 0;

    if (ValueNumber::hasMore(vn.index)) {
        boost::hash_combine(seed, vn.index);
        boost::hash_combine(seed, vn.size);

        for (size_t i = 0, e = vn.size; i != e; ++i)
            boost::hash_combine(seed, vn.more[i]);

        return seed;
    }

    boost::hash_combine(seed, vn.index);
    boost::hash_combine(seed, vn.op1);
    boost::hash_combine(seed, vn.op2);
    boost::hash_combine(seed, vn.op3);

    return seed;
}

} // namespace anydsl
