#ifndef ANYDSL_SUPPORT_UNIVERSE_H
#define ANYDSL_SUPPORT_UNIVERSE_H

#include "anydsl/air/type.h"

namespace anydsl {

class Universe {
public:

    Universe();
    ~Universe();

#define ANYDSL_U_TYPE(T) PrimType* get_##T() const { return T##_; }
#define ANYDSL_F_TYPE(T) PrimType* get_##T() const { return T##_; }
#include "anydsl/tables/primtypetable.h"

    PrimType* get(PrimTypeKind kind) const { 
        size_t i = kind - PrimType_u1;
        assert(0 <= i && i < (size_t) Num_PrimTypes); 
        return primTypes_[kind - PrimType_u1];
    }

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

inline Universe& universe() { static Universe uni; return uni; }

} // namespace anydsl

#endif // ANYDSL_SUPPORT_UNIVERSE_H
