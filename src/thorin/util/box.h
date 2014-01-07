#ifndef THORIN_UTIL_BOX_H
#define THORIN_UTIL_BOX_H

#include <cstring>

#include "thorin/util/assert.h"
#include "thorin/util/types.h"

namespace thorin {

union Box {
public:
    Box()      { reset(); }
#define THORIN_ALL_TYPE(T) Box(T val) { reset(); T##_ = val; }
#include "thorin/tables/primtypetable.h"
    Box(bool val) { reset(); bool_ = val; }

    bool operator == (const Box& other) const { return bcast<uint64_t, Box>(*this) == bcast<uint64_t, Box>(other); }
    template <typename T> inline T get() { THORIN_UNREACHABLE; } 
#define THORIN_ALL_TYPE(T) \
    T get_##T() const { return T##_; }
#include "thorin/tables/primtypetable.h"
    bool get_bool() const { return bool_; }

private:
    void reset() { memset(this, 0, sizeof(Box)); }

#define THORIN_ALL_TYPE(T) T T##_;
#include "thorin/tables/primtypetable.h"
    bool bool_;
};

static_assert(sizeof(Box) == sizeof(uint64_t), "Box has incorrect size in bytes");

#define THORIN_ALL_TYPE(T) template <> inline  T Box::get<T>() { return T##_; }
#include "thorin/tables/primtypetable.h"

inline size_t hash_value(Box box) { return hash_value(bcast<u64, Box>(box)); }

}

#endif
