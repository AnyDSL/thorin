#include "anydsl/air/use.h"

#include "anydsl/air/def.h"

namespace anydsl {

Use::Use(Def* def, AIRNode* parent, const std::string& debug /*= ""*/)
    : AIRNode(Index_Use, debug)
    , def_(def) 
    , parent_(parent)
{
    def->registerUse(this);
}

} // namespace anydsl
