#ifndef THORIN_ANALYSES_BTA_H
#define THORIN_ANALYSES_BTA_H

#include <iostream>

namespace thorin {

/* Forward declarations */
struct World;

/// \brief Represents a value in the abstract domain.
///
/// The lattice for our analysis is
///
///   D
///   |
///   S
struct LV {
    enum Type { Static = 0, Dynamic = 1 };
    Type type : 1;

    LV() : type(Static) { }
    LV(Type const t) : type(t) { }

    LV join(LV const other) const;

    friend std::string to_string(LV const lv);
    friend std::ostream & operator<<(std::ostream &os, LV const lv) { return os << to_string(lv); }
    void dump(std::ostream &os) const { os << *this << std::endl; }
    void dump() const { dump(std::cerr); }

    bool operator==(LV const other) const { return type == other.type; }
    bool operator!=(LV const other) const { return not (*this == other); }
    bool operator< (LV const other) const { return type <  other.type; }
};

void bta(World&);

}

#endif
