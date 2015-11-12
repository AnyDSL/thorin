#include "thorin/be/thorin.h"

#include "thorin/lambda.h"
#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/printer.h"

namespace thorin {

class CodeGen : public Printer {
public:
    CodeGen(std::ostream& ostream)
        : Printer(ostream)
    {}

    std::ostream& emit_type_elems(Type);
    std::ostream& emit_head(const Lambda*);
    std::ostream& emit_jump(const Lambda*);
};

std::ostream& emit_head(const Lambda* lambda, std::ostream& out) {
    out << lambda;
    stream_type_vars(out, lambda->type());
    stream_list(out, [&](const Param* param) { streamf(out, "% %", param->type(), param); }, lambda->params(), "(", ")");

    if (lambda->is_external())
        out << " extern ";
    if (lambda->cc() == CC::Device)
        out << " device ";

    return out << up << endl;
}

std::ostream& emit_jump(const Lambda* lambda, std::ostream& out) {
    if (!lambda->empty()) {
        out << lambda->to();
        stream_list(out, [&](Def def) { out << def; }, lambda->args(), " ", "");
    }
    return out << down << endl;
}

std::ostream& Scope::stream(std::ostream& out) const {
    auto schedule = schedule_smart(*this);
    for (auto& block : schedule) {
        auto lambda = block.lambda();
        if (lambda->intrinsic() != Intrinsic::EndScope) {
            bool indent = lambda != entry();
            if (indent)
                out << up;
            out << endl;
            emit_head(lambda);
            for (auto primop : block)
                primop->stream_assignment(out);

            emit_jump(lambda);
            if (indent)
                out << down;
        }
    }
    return out << endl;
}

std::ostream& World::stream(std::ostream& out) const {
    out << "module '" << name() << "'\n\n";

    for (auto primop : primops()) {
        if (auto global = primop->isa<Global>())
            global->stream_assignment(out);
    }

    Scope::for_each<false>(*this, [&] (const Scope& scope) { scope.stream(out); });
    return out;
}

void Scope::write_thorin(const char* filename) const { std::ofstream file(filename); stream(file); }
void World::write_thorin(const char* filename) const { std::ofstream file(filename); stream(file); }

void Scope::thorin() const {
    auto filename = world().name() + "_" + entry()->unique_name() + ".thorin";
    write_thorin(filename.c_str());
}

void World::thorin() const {
    auto filename = name() + ".thorin";
    write_thorin(filename.c_str());
}

//------------------------------------------------------------------------------

}
