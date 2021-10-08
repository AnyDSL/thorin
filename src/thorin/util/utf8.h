#ifndef THORIN_UTIL_UTF8_H
#define THORIN_UTIL_UTF8_H

#include <array>
#include <optional>

#include "thorin/util/types.h"

// remove when migrating to C++20
using char8_t = uint8_t;

namespace thorin::utf8 {

static constexpr size_t Max = 4;
static constexpr char32_t BOM = 0xfeff;

/// Returns the expected number of bytes for an utf8 char sequence by inspecting the first byte.
/// Retuns @c 0 if invalid.
size_t num_bytes(char8_t c);

/// Append @p b to @p c for converting utf-8 to a code.
inline char32_t append(char32_t c, char32_t b) { return (c << 6_u32) | (b & 0b00111111_u32); }

/// Get relevant bits of first utf-8 byte @p c of a @em multi-byte sequence consisting of @p num bytes.
inline char32_t first(char32_t c, char32_t num) { return c & (0b00011111_u32 >> (num - 2_u32)); }

/// Is the 2nd, 3rd, or 4th byte of an utf-8 byte sequence valid?
inline std::optional<char8_t> is_valid(char8_t c) {
    return (c & 0b11000000_u8) == 0b10000000_u8 ? (c & 0b00111111_u8) : std::optional<char8_t>();
}

}

#endif
