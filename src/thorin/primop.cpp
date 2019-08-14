#include "thorin/primop.h"

#include "thorin/config.h"
#include "thorin/world.h"
#include "thorin/util.h"
#include "thorin/util/array.h"

namespace thorin {

//------------------------------------------------------------------------------

/*
 * constructors
 */

Cmp::Cmp(CmpTag tag, const Def* lhs, const Def* rhs, const Def* dbg)
    : BinOp((NodeTag) tag, rebuild, lhs->world().type_bool(), lhs, rhs, dbg)
{}

Known::Known(const Def* def, const Def* dbg)
    : PrimOp(Node_Known, rebuild, def->world().type_bool(), {def}, 0, dbg)
{}

SizeOf::SizeOf(const Def* def, const Def* dbg)
    : PrimOp(Node_SizeOf, rebuild, def->world().type_qs32(), {def}, 0, dbg)
{}

/*
Assembly::Assembly(const Def* type, Defs inputs, std::string asm_template, ArrayRef<std::string> output_constraints, ArrayRef<std::string> input_constraints, ArrayRef<std::string> clobbers, Flags flags, const Def* dbg)
    : MemOp(Node_Assembly, rebuild, type, inputs, uint64_t(flags), dbg)
{
    new (&extra<Extra>()) Extra();
    extra<Extra>().asm_template_ = asm_template;
    extra<Extra>().output_constraints_ = output_constraints;
    extra<Extra>().input_constraints_ = input_constraints;
    extra<Extra>().clobbers_ = clobbers;
}

Assembly::~Assembly() { (&extra<Extra>())->~Extra(); }
*/

//------------------------------------------------------------------------------

/*
 * rebuild
 */

// do not use any of PrimOp's type getters - during import we need to derive types from 't' in the new world 'to'

const Def* Alloc  ::rebuild(const Def*  , World& to, const Def* t, Defs ops, const Def* dbg) { return to.alloc(t->as<Sigma>()->op(1)->as<PtrType>()->pointee(), ops[0], dbg); }
const Def* ArithOp::rebuild(const Def* d, World& to, const Def*  , Defs ops, const Def* dbg) { return to.arithop(d->as<ArithOp>()->arithop_tag(), ops[0], ops[1], dbg); }
const Def* Bitcast::rebuild(const Def*  , World& to, const Def* t, Defs ops, const Def* dbg) { return to.bitcast(t, ops[0], dbg); }
const Def* Cast   ::rebuild(const Def*  , World& to, const Def* t, Defs ops, const Def* dbg) { return to.cast(t, ops[0], dbg); }
const Def* Cmp    ::rebuild(const Def* d, World& to, const Def*  , Defs ops, const Def* dbg) { return to.cmp(d->as<Cmp>()->cmp_tag(), ops[0], ops[1], dbg); }
const Def* Global ::rebuild(const Def* d, World& to, const Def*  , Defs ops, const Def* dbg) { return to.global(ops[0], d->as<Global>()->is_mutable(), dbg); }
const Def* Hlt    ::rebuild(const Def*  , World& to, const Def*  , Defs ops, const Def* dbg) { return to.hlt(ops[0], dbg); }
const Def* Known  ::rebuild(const Def*  , World& to, const Def*  , Defs ops, const Def* dbg) { return to.known(ops[0], dbg); }
const Def* Run    ::rebuild(const Def*  , World& to, const Def*  , Defs ops, const Def* dbg) { return to.run(ops[0], dbg); }
const Def* LEA    ::rebuild(const Def*  , World& to, const Def*  , Defs ops, const Def* dbg) { return to.lea(ops[0], ops[1], dbg); }
const Def* Load   ::rebuild(const Def*  , World& to, const Def*  , Defs ops, const Def* dbg) { return to.load(ops[0], ops[1], dbg); }
const Def* Select ::rebuild(const Def*  , World& to, const Def*  , Defs ops, const Def* dbg) { return to.select(ops[0], ops[1], ops[2], dbg); }
const Def* SizeOf ::rebuild(const Def*  , World& to, const Def*  , Defs ops, const Def* dbg) { return to.size_of(ops[0]->type(), dbg); }
const Def* Slot   ::rebuild(const Def*  , World& to, const Def* t, Defs ops, const Def* dbg) { return to.slot(t->as<Sigma>()->op(1)->as<PtrType>()->pointee(), ops[0], dbg); }
const Def* Store  ::rebuild(const Def*  , World& to, const Def*  , Defs ops, const Def* dbg) { return to.store(ops[0], ops[1], ops[2], dbg); }
const Def* Variant::rebuild(const Def*  , World& to, const Def* t, Defs ops, const Def* dbg) { return to.variant(t->as<VariantType>(), ops[0], dbg); }

//const Def* Assembly::rebuild(const Def* d, World& to, const Def* t, Defs ops, const Def* dbg) {
    //auto asm_ = d->as<Assembly>();
    //return to.assembly(t, ops, asm_->asm_template(), asm_->output_constraints(), asm_->input_constraints(), asm_->clobbers(), asm_->flags(), dbg);
//}

//------------------------------------------------------------------------------

/*
 * op_name
 */

const char* Def::op_name() const {
    switch (tag()) {
#define THORIN_NODE(op, abbr) case Node_##op: return #abbr;
#include "thorin/tables/nodetable.h"
        default: THORIN_UNREACHABLE;
    }
}

const char* ArithOp::op_name() const {
    switch (tag()) {
#define THORIN_ARITHOP(op) case ArithOp_##op: return #op;
#include "thorin/tables/arithoptable.h"
        default: THORIN_UNREACHABLE;
    }
}

const char* Cmp::op_name() const {
    switch (tag()) {
#define THORIN_CMP(op) case Cmp_##op: return #op;
#include "thorin/tables/cmptable.h"
        default: THORIN_UNREACHABLE;
    }
}

const char* Global::op_name() const { return is_mutable() ? "global_mutable" : "global_immutable"; }

//------------------------------------------------------------------------------

/*
 * stream
 */

std::ostream& PrimOp::stream(std::ostream& os) const {
    if (is_const(this)) {
        if (num_ops() == 0)
            return streamf(os, "{} {}", op_name(), type());
        else
            return streamf(os, "({} {} {})", type(), op_name(), stream_list(ops(), [&](const Def* def) { os << def; }));
    } else
        return os << unique_name();
}

std::ostream& Global::stream(std::ostream& os) const { return os << unique_name(); }

/*
std::ostream& Assembly::stream_assignment(std::ostream& os) const {
    streamf(os, "{} {} = asm \"{}\"", type(), unique_name(), asm_template());
    stream_list(os, output_constraints(), [&](const auto& output_constraint) { os << output_constraint; }, " : (", ")");
    stream_list(os,  input_constraints(), [&](const auto&  input_constraint) { os <<  input_constraint; }, " : (", ")");
    stream_list(os,           clobbers(), [&](const auto&           clobber) { os <<           clobber; }, " : (", ") ");
    return stream_list(os,         ops(), [&](const Def*                def) { os <<               def; },    "(", ")") << endl;
}
*/

}
