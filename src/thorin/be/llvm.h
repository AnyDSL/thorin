#ifndef THORIN_BE_LLVM_H
#define THORIN_BE_LLVM_H

#include "thorin/def.h"
#include "thorin/util/assert.h"

namespace llvm {
    class IRBuilderBase;
    class Type;
    class Value;
    class Module;
}

namespace thorin {

class Type;

class World;

class EmitHook {
public:
    virtual ~EmitHook() {}

    virtual void assign(llvm::IRBuilderBase* builder, llvm::Module* module) {}
    virtual llvm::Value* emit(Def) { THORIN_UNREACHABLE; }
    virtual llvm::Type* map(const Type*) { THORIN_UNREACHABLE; }
};

#ifdef LLVM_SUPPORT
void emit_llvm(World& world, EmitHook& hook);
#else
inline void emit_llvm(World& world, EmitHook& hook) {}
#endif
inline void emit_llvm(World& world) { EmitHook hook; emit_llvm(world, hook); }

} // namespace thorin

#endif
