#include "thorin/be/codegen.h"

namespace thorin::shady {

extern "C" {
#include <shady/ir.h>
}

class CodeGen : public thorin::CodeGen {
public:
    CodeGen(World&, Cont2Config&, bool debug);

    void emit_stream(std::ostream& stream) override;
    const char* file_ext() const override { return ".shady"; }

    shady::Type* convert(const Type*);
protected:
    void emit(const Scope& scope);
    //void emit_epilogue(Continuation*, BasicBlockBuilder* bb);
    //shady::Node* emit(const Def* def, BasicBlockBuilder* bb);
    //std::vector<SpvId> emit_builtin(const Continuation*, const Continuation*, BasicBlockBuilder*);

    //SpvId get_codom_type(const Continuation* fn);
    shady::IrArena* arena = nullptr;
    std::vector<std::pair<shady::Node*, shady::Node*>> top_level;

    Continuation* entry_ = nullptr;
    TypeMap<std::unique_ptr<shady::Type*>> types_;
    DefMap<shady::Node*> defs_;
    const Cont2Config& kernel_config_;

};

}
