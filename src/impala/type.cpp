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

bool NoRet::equal(const Type* t) const {
    return t->isa<NoRet>();
}

size_t NoRet::hash() const {
    return boost::hash_value(Token::TYPE_noret);
}

Pi::Pi(anydsl::ArrayRef<const Type*> elems, const Type* retType) 
    : elems_(elems)
    , retType_(retType)
{}

size_t Pi::hash() const {
    size_t seed = 0;
    boost::hash_combine(seed, numElems());

    for_all (elem, elems())
        boost::hash_combine(seed, elem);

    boost::hash_combine(seed, retType_);

    return seed;
}

bool Pi::equal(const Type* other) const {
    if (const Pi* pi = other->isa<Pi>()) {
        if (numElems() != pi->numElems() || retType_ != pi->retType())
            return false;

        bool result = true;

        for (size_t i = 0; i < numElems() && result; ++i)
            result &= elems()[i] == pi->elems()[i];

        return result;
    }

    return false;
}

//------------------------------------------------------------------------------

TypeTable::TypeTable() 
    : type_error_((*types_.insert(new TypeError()).first)->as<TypeError>())
    , type_void_ ((*types_.insert(new Void()).first)->as<Void>())
    , noret_((*types_.insert(new NoRet()).first)->as<NoRet>())
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

const Pi* TypeTable::pi(anydsl::ArrayRef<const Type*> elems, const Type* retType) {
    const Pi* pi = new Pi(elems, retType);
    TypeSet::iterator i = types_.find(pi);

    if (i == types_.end()) {
        types_.insert(pi);
        return pi;
    }

    delete pi;
    return (*i)->as<Pi>();
}

} // namespace impala
