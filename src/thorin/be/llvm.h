#ifndef THORIN_BE_LLVM_H
#define THORIN_BE_LLVM_H

namespace thorin {

class World;

#ifdef LLVM_SUPPORT
void emit_llvm(World& world);
#else
inline void emit_llvm(World& world) {}
#endif

} // namespace thorin

#endif
