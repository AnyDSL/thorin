#include "shady.h"

#include "thorin/analyses/scope.h"
#include "thorin/transform/structurize.h"

namespace thorin::shady {

CodeGen::CodeGen(thorin::World& world, Cont2Config& kernel_config, bool debug)
        : thorin::CodeGen(world, debug), kernel_config_(kernel_config)
{}

void CodeGen::emit_stream(std::ostream& out) {
    assert(top_level.empty());

    structure_loops(world());
    structure_flow(world());

    auto config = shady::IrConfig {
        .check_types = true,
    };
    arena = shady::new_arena(config);

    Scope::for_each(world(), [&](const Scope& scope) { emit(scope); });

    // build root node with the top level stuff that got emitted
    auto defs = std::vector<shady::Node*>(top_level.size(), nullptr);
    auto vars = std::vector<shady::Node*>(top_level.size(), nullptr);
    for (size_t i = 0; i < top_level.size(); i++) {
        defs[i] = top_level[i].first;
        vars[i] = top_level[i].second;
    }
    auto root = shady::root(arena, (shady::Root) {
        .variables = shady::nodes(arena, top_level.size(), const_cast<const shady::Node**>(vars.data())),
        .definitions = shady::nodes(arena, top_level.size(), const_cast<const shady::Node**>(defs.data())),
    });

    shady::print_node(root);

    out << "todo";

    shady::destroy_arena(arena);
    arena = nullptr;
    top_level.clear();
}

void CodeGen::emit(const thorin::Scope& scope) {
    entry_ = scope.entry();
    assert(entry_->is_returning());

    assert(false && "TODO");
}

shady::Type* CodeGen::convert(const Type *) {
    return nullptr;
}

}
