#ifndef ANYDSL2_BE_LLVM_H
#define ANYDSL2_BE_LLVM_H

#include "anydsl2/util/assert.h"

namespace llvm {
    class IRBuilderBase;
    class Type;
    class Value;
    class Module;
}

namespace anydsl2 {

class Def;
class Type;

class World;

namespace be_llvm {

class EmitHook {
public:

    virtual ~EmitHook() {}

    virtual void assign(llvm::IRBuilderBase* builder, llvm::Module* module) {}
    virtual llvm::Value* emit(const Def*) { ANYDSL2_UNREACHABLE; }
    virtual llvm::Type* map(const Type*) { ANYDSL2_UNREACHABLE; }
};

void emit(const World& world, EmitHook& hook);
inline void emit(const World& world) { EmitHook hook; emit(world, hook); }

} // namespace anydsl2
} // namespace be_llvm

#endif
