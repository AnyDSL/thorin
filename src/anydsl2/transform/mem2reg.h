#ifndef ANYDSL2_TRANSFORM_MEM2REG_H
#define ANYDSL2_TRANSFORM_MEM2REG_H

namespace anydsl2 {

/**
 * Tries to remove \p Load%s, \p Store%s, and \p Slot%s in favor of \p Param%s and arguments of function calls.
 * Moreover, superflous \p Enter and \p Leave \p PrimOp%s are removed.
 * \attention { Currently, this transformation only works when in CFF. }
 */
void mem2reg(World&);

} // namespace anydsl2

#endif // ANYDSL2_TRANSFORM_MEM2REG_H
