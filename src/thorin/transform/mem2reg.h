#ifndef THORIN_TRANSFORM_MEM2REG_H
#define THORIN_TRANSFORM_MEM2REG_H

namespace thorin {

template<bool> class ScopeBase;
class World;

/**
 * Tries to remove \p Load%s, \p Store%s, and \p Slot%s in favor of \p Param%s and arguments of function calls.
 * \attention { Currently, this transformation only works when in CFF. }
 */
void mem2reg(World&);
void mem2reg(const ScopeBase<true>&);

} // namespace thorin

#endif // THORIN_TRANSFORM_MEM2REG_H
