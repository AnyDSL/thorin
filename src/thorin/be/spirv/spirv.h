#ifndef THORIN_SPIRV_H
#define THORIN_SPIRV_H

#include <thorin/analyses/schedule.h>
#include "thorin/be/backends.h"

namespace thorin::spirv {

struct SpvSectionBuilder;
struct SpvBasicBlockBuilder;
struct SpvFnBuilder;
struct SpvFileBuilder;
struct SpvId { uint32_t id; };

class CodeGen : public thorin::CodeGen {
public:
    CodeGen(World&, Cont2Config&, bool debug);

    void emit(std::ostream& stream) override;
protected:
    SpvId convert(const Type*);
    void emit(const Scope& scope);
    void emit_epilogue(Continuation*, SpvBasicBlockBuilder* bb);
    SpvId emit(const Def* def, SpvBasicBlockBuilder* bb);

    SpvId get_codom_type(const Continuation* fn);

    SpvFileBuilder* builder_ = nullptr;
    Continuation* entry_ = nullptr;
    SpvFnBuilder* current_fn_ = nullptr;
    Scheduler scheduler_;
    TypeMap<SpvId> types_;
    DefMap<SpvId> defs_;
};

}

#endif //THORIN_SPIRV_H
