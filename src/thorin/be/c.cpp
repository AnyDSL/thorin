#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/top_level_scopes.h"
#include "thorin/util/autoptr.h"
#include "thorin/util/printer.h"

namespace thorin {

class CodeGen : public Printer {
public:
    CodeGen(World& world, std::ostream& stream)
        : Printer(stream)
        , world_(world)
    {}

    void emit();
    void emit(Def def);

private:
    World& world_;
};

void CodeGen::emit() {
    auto scopes = top_level_scopes(world_);
}

void CodeGen::emit(Def def) {
}

//------------------------------------------------------------------------------

void emit_c(World& world, std::ostream& stream) { CodeGen(world, stream).emit(); }

//------------------------------------------------------------------------------

}
