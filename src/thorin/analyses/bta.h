#ifndef THORIN_ANALYSES_BTA_H
#define THORIN_ANALYSES_BTA_H

#include <iostream>
#include <vector>
#include "thorin/def.h"

namespace thorin {

/* Forward declarations */
struct World;
struct Lambda;
struct Param;
struct PrimOp;

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
    bool isTop() const { return Top.type == type; }
    bool isBot() const { return Bot.type == type; }

    friend std::string to_string(LV const lv);
    friend std::ostream & operator<<(std::ostream &os, LV const lv) { return os << to_string(lv); }
    void dump(std::ostream &os) const { os << *this << std::endl; }
    void dump() const { dump(std::cerr); }

    bool operator==(LV const other) const { return type == other.type; }
    bool operator!=(LV const other) const { return not (*this == other); }
    bool operator< (LV const other) const { return type <  other.type; }

    static LV const Top;
    static LV const Bot;
};

struct BTA {
    void run(World &world);
    LV   get(DefNode const *def);

    private:
    void visit(DefNode const *def);
    void visit(Lambda  const *def);
    void visit(Param   const *def);
    void visit(PrimOp  const *def);

    bool update(DefNode const *def, LV const lv);

    std::vector<DefNode const *> worklist;
    DefMap<LV> LatticeValues;
};

void bta(World&);

}

#endif
