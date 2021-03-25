#ifndef DOT_DUMP_H
#define DOT_DUMP_H

#include "thorin/world.h"

namespace thorin {

/// Outputs the raw thorin IR as a graph without performing any scope or scheduling analysis
struct DotPrinter {
    DotPrinter(World& world, const char* filename = "world.dot") : world(world) {
        file = std::ofstream(filename);
    }

    private:
        void dump_def(const Def* def);
        void dump_def_generic(const Def* def, const char* color, const char* shape);
        void dump_literal(const Literal* cont);
        void dump_primop(const PrimOp* cont);
        void dump_continuation(const Continuation* cont);

    #define up ""
    #define down ""
    #define endl "\n"

    public:
        void print() {
            file << "digraph " << world.name() << " {" << up;
            file << endl << "bgcolor=transparent;";
            for (Continuation* continuation : world.continuations()) {
                // Ignore those if they are not referenced elsewhere...
                if (continuation == world.branch() || continuation == world.end_scope())
                    continue;
                dump_def(continuation);
            }
            file << down << endl << "}" << endl;
        }

    private:
        thorin::World& world;

        DefSet done;
        std::ofstream file;
};

void DotPrinter::dump_def(const Def* def) {
    if (done.contains(def))
        return;

    if (def->isa_continuation())
        dump_continuation(def->as_continuation());
    else if (def->isa<Literal>())
        dump_literal(def->as<Literal>());
    else if (def->isa<PrimOp>())
        dump_primop(def->as<PrimOp>());
    else if (def->isa<Param>())
        dump_def_generic(def, "grey", "oval");
    else
        dump_def_generic(def, "red", "star");
}

void DotPrinter::dump_def_generic(const Def* def, const char* color, const char* shape) {
    file << endl << def->unique_name() << " [" << up;

    file << endl << "label = \"";

    file << def->unique_name() << " : " << def->type()->to_string();

    file << "\";";

    file << endl << "shape = " << shape << ";";
    file << endl << "color = " << color << ";";

    file << down << endl << "]";

    done.emplace(def);

    for (size_t i = 0; i < def->num_ops(); i++) {
        const auto& op = def->op(i);
        dump_def(op);
        file << endl << def->unique_name() << " -> " << op->unique_name() << " [arrowhead=vee,label=\"o" << i << "\",fontsize=8,fontcolor=grey];";
    }
}

void DotPrinter::dump_literal(const Literal* def) {
    file << endl << def->unique_name() << " [" << up;

    file << endl << "label = \"";
    file << def->to_string();
    file << "\";";

    file << endl << "style = dotted;";

    file << down << endl << "]";

    done.emplace(def);

    assert(def->num_ops() == 0);
}

void DotPrinter::dump_primop(const PrimOp* def) {
    file << endl << def->unique_name() << " [" << up;

    file << endl << "label = \"";

    file << def->op_name();

    auto variant = def->isa<Variant>();
    auto variant_extract = def->isa<VariantExtract>();
    if (variant || variant_extract )
        file << "(" << (variant ? variant->index() : variant_extract->index()) << ")";

    file << "\";";

    file << endl << "color = darkseagreen1;";
    file << endl << "style = filled;";

    file << down << endl << "]";

    done.emplace(def);

    for (size_t i = 0; i < def->num_ops(); i++) {
        const auto& op = def->op(i);
        dump_def(op);
        file << endl << def->unique_name() << " -> " << op->unique_name() << " [arrowhead=vee,label=\"o" << i << "\",fontsize=8,fontcolor=grey];";
    }
}

void DotPrinter::dump_continuation(const Continuation* cont) {
    done.emplace(cont);
    auto intrinsic = cont->intrinsic();
    file << endl << cont->unique_name() << " [" << up;

    file << endl << "label = \"";
    if (cont->is_exported())
        file << "[extern]\\n";
    auto name = cont->name();
    if (!cont->is_exported())
        name = cont->unique_name();
    file << name << "(";
    for (size_t i = 0; i < cont->num_params(); i++) {
        file << cont->param(i)->type()->to_string() << (i + 1 == cont->num_params() ? "" : ", ");
    }
    file << ")";
    file << "\";";

    file << endl << "shape = rectangle;";
    if (intrinsic != Intrinsic::None) {
        file << endl << "color = lightblue;";
        file << endl << "style = filled;";
    }
    if (cont->is_exported()) {
        file << endl << "color = pink;";
        file << endl << "style = filled;";
    }

    file << down << endl << "]";

    int x = 1;
    switch (intrinsic) {
        case Intrinsic::SCFLoopHeader:
            dump_def(cont->op(1));
            file << endl << cont->unique_name() << " -> " << cont->op(1)->unique_name() << " [arrowhead=none];";
            x = 2;
        case Intrinsic::SCFLoopContinue:
        case Intrinsic::SCFLoopMerge:
            dump_def(cont->op(0));
            file << endl << cont->unique_name() << " -> " << cont->op(0)->unique_name() << " [arrowhead=none];";
            for (size_t i = x; i < cont->num_ops(); i++) {
                auto op = cont->op(i);
                dump_def(op);
                file << endl << cont->unique_name() << " -> " << op->unique_name() << " [arrowhead=normal];";
            }
            return;

        default: break;
    }

    if (auto callee_cont = cont->callee()->isa_continuation()) {
        switch (callee_cont->intrinsic()) {
            case Intrinsic::Branch: {
                auto condition = cont->arg(0);
                dump_def(condition);
                file << endl << condition->unique_name() << " -> " << cont->unique_name()
                     << " [arrowhead=onormal,label=\"condition\",fontsize=8,fontcolor=grey];";
                auto if_true = cont->arg(1);
                dump_def(if_true);
                file << endl << cont->unique_name() << " -> " << if_true->unique_name()
                     << " [arrowhead=normal,label=\"if_true\",fontsize=8,fontcolor=grey];";
                auto if_false = cont->arg(2);
                dump_def(if_false);
                file << endl << cont->unique_name() << " -> " << if_false->unique_name()
                     << " [arrowhead=normal,label=\"if_false\",fontsize=8,fontcolor=grey];";
                return;
            }
            default:
                break;
        }
    }

    for (size_t i = 0; i < cont->num_args(); i++) {
        auto arg = cont->arg(i);
        dump_def(arg);

        if (cont->callee()->uses().size() > 1)
            file << endl << arg->unique_name() << " -> " << cont->callee()->unique_name() << " [arrowhead=onormal,label=\"a" << i << " from " << cont->unique_name() << "\",fontsize=8,fontcolor=grey];";
        else
            file << endl << arg->unique_name() << " -> " << cont->callee()->unique_name() << " [arrowhead=onormal,label=\"a" << i << "\",fontsize=8,fontcolor=grey];";
    }

    switch (intrinsic) {
        // We don't care about the params for these, or the callee
        case Intrinsic::Match:
        case Intrinsic::Branch:
            return;
        default:
            break;
    }

    for (size_t i = 0; i < cont->num_params(); i++) {
        auto param = cont->param(i);
        dump_def(param);
        file << endl << param->unique_name() << " -> " << cont->unique_name() << " [arrowhead=none,label=\"p" << i << "\",fontsize=8,fontcolor=grey];";
    }
    dump_def(cont->callee());
    file << endl << cont->unique_name() << " -> " << cont->callee()->unique_name() << " [arrowhead=normal];";
}

void dump_dot(World& world) {
    auto p = DotPrinter(world);
    p.print();
}

}

#endif //DOT_DUMP_H
