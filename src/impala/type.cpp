#include "impala/type.h"

#include <iterator>
#include <boost/functional/hash.hpp>

#include "anydsl/util/cast.h"
#include "anydsl/util/for_all.h"

using anydsl::dcast;

namespace impala {

bool PrimType::equal(const Type* t) const {
    if (const PrimType* p = t->isa<PrimType>())
        return kind() == p->kind();

    return false;
}

size_t PrimType::hash() const {
    return boost::hash_value(kind());
}

bool TypeError::equal(const Type* t) const {
    return t->isa<TypeError>();
}

size_t TypeError::hash() const {
    return boost::hash_value(Token::END_OF_FILE);
}

bool Void::equal(const Type* t) const {
    return t->isa<Void>();
}

size_t Void::hash() const {
    return boost::hash_value(Token::TYPE_void);
}

Pi::Pi(const Type* const* begin, const Type* const* end, const Type* retType) 
    : numArgs_(std::distance(begin, end))
    , args_(new const Type*[numArgs_])
    , retType_(retType)
{
    for (size_t i = 0; i != numArgs_; ++i)
        args_[i] = begin[i];
}

Pi::~Pi() {
    delete[] args_;
}

size_t Pi::hash() const {
    size_t seed = 0;
    boost::hash_combine(seed, numArgs_);

    for (size_t i = 0; i < numArgs_; ++i)
        boost::hash_combine(seed, args_[i]);

    boost::hash_combine(seed, retType_);

    return seed;
}

bool Pi::equal(const Type* other) const {
    if (const Pi* pi = other->isa<Pi>()) {
        if (numArgs_ != pi->numArgs() || retType_ != pi->retType())
            return false;

        bool result = true;

        for (size_t i = 0; i < numArgs_ && result; ++i)
            result &= args()[i] == pi->args()[i];

        return result;
    }

    return false;
}

//------------------------------------------------------------------------------

TypeTable::TypeTable() 
    : type_error_((*types_.insert(new TypeError()).first)->as<TypeError>())
    , type_void_ ((*types_.insert(new Void()).first)->as<Void>())
#define IMPALA_TYPE(itype, atype) \
    , itype##_((*types_.insert(new PrimType(PrimType::TYPE_##itype)).first)->as<PrimType>())
#include "impala/tokenlist.h"
{}

TypeTable::~TypeTable() {
    for_all (type, types_)
        delete type;
}

const PrimType* TypeTable::type(PrimType::Kind kind) {
    switch (kind) {
#define IMPALA_TYPE(itype, atype) \
        case PrimType::TYPE_##itype: return itype##_;
#include "impala/tokenlist.h"
        default: ANYDSL_UNREACHABLE;
    }
}

const Pi* TypeTable::pi(const Type* const* begin, const Type* const* end, const Type* retType) {
    const Pi* pi = new Pi(begin, end, retType);
    TypeSet::iterator i = types_.find(pi);

    if (i == types_.end()) {
        types_.insert(pi);
        return pi;
    }

    delete pi;
    return (*i)->as<Pi>();
}

} // namespace impala
