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
    Box(  int8_t val) { reset();   int8_t_ = val; }
    Box( int16_t val) { reset();  int16_t_ = val; }
    Box( int32_t val) { reset();  int32_t_ = val; }
    Box( int64_t val) { reset();  int64_t_ = val; }
    Box( uint8_t val) { reset();  uint8_t_ = val; }
    Box(uint16_t val) { reset(); uint16_t_ = val; }
    Box(uint32_t val) { reset(); uint32_t_ = val; }
    Box(uint64_t val) { reset(); uint64_t_ = val; }
    Box(float    val) { reset(); float_    = val; }
    Box(double   val) { reset(); double_   = val; }

    bool operator == (const Box& other) const { return bcast<uint64_t, Box>(*this) == bcast<uint64_t, Box>(other); }
    template <typename T> inline T get() { THORIN_UNREACHABLE; } 
#define THORIN_ALL_TYPE(T) \
    T get_##T() const { return T##_; }
#include "thorin/tables/primtypetable.h"

private:
    void reset() { memset(this, 0, sizeof(Box)); }

#define THORIN_ALL_TYPE(T) T T##_;
#include "thorin/tables/primtypetable.h"
      int8_t  int8_t_;  int16_t  int16_t_;  int32_t  int32_t_;  int64_t  int64_t_;
     uint8_t uint8_t_; uint16_t uint16_t_; uint32_t uint32_t_; uint64_t uint64_t_;
     float float_; double double_;
};

static_assert(sizeof(Box) == sizeof(uint64_t), "Box has incorrect size in bytes");

#define THORIN_ALL_TYPE(T) template <> inline  T Box::get<T>() { return T##_; }
#include "thorin/tables/primtypetable.h"

inline size_t hash_value(Box box) { return hash_value(bcast<u64, Box>(box)); }

}

#endif
