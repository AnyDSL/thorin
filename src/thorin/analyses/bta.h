#ifndef THORIN_ANALYSES_BTA_H
#define THORIN_ANALYSES_BTA_H

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

    LV join(LV other) const;
};

void bta(World&);

}

#endif
