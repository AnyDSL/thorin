#ifndef ANYDSL_UNIVERSE_H
#define ANYDSL_UNIVERSE_H

#include "anydsl/air/type.h"

namespace anydsl {

class Universe {
public:

    Universe();
    ~Universe();

#define ANYDSL_U_TYPE(T) \
    PrimType* get_##T() const { return T##_; } \
    //PrimType* get(PrimTypeKind kind) const { return T##_; }
#define ANYDSL_F_TYPE(T) PrimType* get_##T() const { return T##_; }
#include "anydsl/tables/primtypetable.h"
//#define ANYDSL_U_TYPE(T) PrimType* get(PrimTypeKind primTypeKind) { return primTypes_
//#include "anydsl/tables/primtypetable.h"

private:

    bool dummy_;

    union {
        struct {
#define ANYDSL_U_TYPE(T) PrimType* T##_;
#define ANYDSL_F_TYPE(T) PrimType* T##_;
#include "anydsl/tables/primtypetable.h"
        };

        PrimType* primTypes_[Num_PrimTypes];
    };
};

} // namespace anydsl

#endif // ANYDSL_UNIVERSE_H
