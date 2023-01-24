#include "thorin/be/codegen.h"
#include "thorin/analyses/schedule.h"
#include "thorin/be/emitter.h"

namespace shady {
extern "C" {
#include <shady/ir.h>
}
}

namespace thorin::shady_be {

// using BB = std::pair<shady::BlockBuilder*, shady::Node*>;

struct BB {
    /// For an entry BB, this will also be the head of the entire function
    shady::Node* head;
    shady::BodyBuilder* builder;
    const shady::Node* terminator;
};

class CodeGen : public thorin::CodeGen, public thorin::Emitter<const shady::Node*, const shady::Type*, BB, CodeGen> {
public:
    CodeGen(World&, Cont2Config&, bool debug);

    void emit_stream(std::ostream& stream) override;
    const char* file_ext() const override { return ".shady"; }

    shady::Node* prepare(const Scope&);
    void prepare(Continuation*, shady::Node*);
    void emit_epilogue(Continuation*);
    // const shady::Node* emit_(const Def* def);
    void finalize(const Scope&);
    void finalize(Continuation*);

    const shady::Type* convert(const Type*);
    const shady::Node* emit_bb(BB&, const Def*);

    bool is_valid(const shady::Node* n) {
        return n;
    }

    const shady::Node* emit_fun_decl(Continuation*);
protected:
    shady::AddressSpace convert_address_space(AddrSpace);
    shady::Node* emit_decl_head(Def*);
    shady::Node* get_decl(Def*);

    using NodeVec = std::vector<const shady::Node*>;

    inline shady::Nodes vec2nodes(NodeVec& vec) {
        return shady::nodes(arena, vec.size(), vec.data());
    }

    shady::IrArena* arena = nullptr;
    shady::Module* module = nullptr;
    //std::vector<shady::Node*> top_level;

    shady::Node* curr_fn;

    const Cont2Config& kernel_config_;

};

}
