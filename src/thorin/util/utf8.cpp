#include "thorin/util/utf8.h"

namespace thorin::utf8 {

size_t num_bytes(char8_t c) {
    if ((c & 0b10000000_u8) == 0b00000000_u8) return 1;
    if ((c & 0b11100000_u8) == 0b11000000_u8) return 2;
    if ((c & 0b11110000_u8) == 0b11100000_u8) return 3;
    if ((c & 0b11111000_u8) == 0b11110000_u8) return 4;
    return 0;
}

}
