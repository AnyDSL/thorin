#include "thorin/util/hash.h"

namespace thorin {
    namespace detail {

        uint16_t HashTableBase::gid_counter_ = 0;

        HashTableBase::HashTableBase()
            : gid_(gid_counter_++)
        {}

    }
}
