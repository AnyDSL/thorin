#include "thorin/be/codegen.h"

namespace shady {
extern "C" {
#include <shady/ir.h>
}
}

namespace thorin::shady_be {

class CodeGen : public thorin::CodeGen {
public:
    CodeGen(World&, Cont2Config&, bool debug);

    void emit_stream(std::ostream& stream) override;
    const char* file_ext() const override { return ".shady"; }

    const shady::Type* convert(const Type*);
protected:
    shady::AddressSpace convert_address_space(AddrSpace);

    void emit(const Scope& scope);
    //void emit_epilogue(Continuation*, BasicBlockBuilder* bb);
    //shady::Node* emit(const Def* def, BasicBlockBuilder* bb);
    //std::vector<SpvId> emit_builtin(const Continuation*, const Continuation*, BasicBlockBuilder*);

    //SpvId get_codom_type(const Continuation* fn);
    shady::IrArena* arena = nullptr;
    std::vector<std::pair<shady::Node*, shady::Node*>> top_level;

    Continuation* entry_ = nullptr;
    TypeMap<const shady::Type*> types_;
    DefMap<const shady::Node*> defs_;
    const Cont2Config& kernel_config_;

};

}
