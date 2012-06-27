#include "anydsl/type.h"

#include "anydsl/literal.h"

namespace anydsl {

//------------------------------------------------------------------------------

PrimType::PrimType(World& world, PrimTypeKind kind)
    : Type(world, (IndexKind) kind, 0)
{
    debug = kind2str(this->kind());
}

//------------------------------------------------------------------------------

NoRet::NoRet(World& world, const Pi* pi)
    : Type(world, Index_NoRet, 1)
{
    setOp(0, pi);
}

void NoRet::dump(Printer& printer) const {
	printer << "noret(";
	pi()->dump(printer);
	printer << ')';
}

//------------------------------------------------------------------------------

CompoundType::CompoundType(World& world, IndexKind index, size_t num)
    : Type(world, index, num)
{}

CompoundType::CompoundType(World& world, IndexKind index, const Type* const* begin, const Type* const* end)
    : Type(world, index, std::distance(begin, end))
{
    size_t x = 0;
    for (const Type* const* i = begin; i != end; ++i, ++x)
        setOp(x, *i);
}

const Type* CompoundType::get(const PrimLit* c) const { 
    anydsl_assert(isInteger(lit2type(c->kind())), "must be an integer constant");
    return get(c->box().get_u64()); 
}

void CompoundType::dump(Printer& printer) const {
	printer << '(';
	printer.dumpOps(ops());
	printer << ')';
}

//------------------------------------------------------------------------------

void PrimType::dump(Printer& printer) const {
	switch(indexKind()) {
#define ANYDSL_U_TYPE(T) case Index_PrimType_##T: printer << #T; return;
#define ANYDSL_F_TYPE(T) ANYDSL_U_TYPE(T)
#include "anydsl/tables/primtypetable.h"
	default:
		ANYDSL_UNREACHABLE;
		break;
	}
}

//------------------------------------------------------------------------------

void Sigma::dump(Printer& printer) const {
	printer << "sigma";
	CompoundType::dump(printer);
}

//------------------------------------------------------------------------------

void Pi::dump(Printer& printer) const {
	printer << "pi";
	CompoundType::dump(printer);
}

} // namespace anydsl
