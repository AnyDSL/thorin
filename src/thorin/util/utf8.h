#ifndef THORIN_UTIL_UTF8_H
#define THORIN_UTIL_UTF8_H

#include "thorin/util/types.h"

#include <array>

namespace thorin {

class UTF8Char {
public:
    static constexpr size_t Max = 4;

    UTF8Char(std::array<uint8_t, Max> chars)
        : chars_(chars)
    {}
    UTF8Char(std::initializer_list<uint8_t> list)
    {
        std::copy(list.begin(), list.end(), chars_.begin());
    }
    UTF8Char(uint32_t u)
        : chars_({
            u8(u           & 0x000000FF_u32),
            u8(u >>  8_u32 & 0x000000FF_u32),
            u8(u >> 16_u16 & 0x000000FF_u32),
            u8(u >> 24_u32)})
    {}

    operator uint32_t() {
        return (u32(chars_[3]) << 24_u32)
            |  (u32(chars_[2]) << 16_u32)
            |  (u32(chars_[1]) <<  8_u32)
            |  (u32(chars_[0])          );
    }

private:
    std::array<uint8_t, Max> chars_;
};

}

#endif
