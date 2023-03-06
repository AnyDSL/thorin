#ifndef DOT_DUMP_H
#define DOT_DUMP_H

#include "thorin/world.h"

namespace thorin {

static auto endl = "\n";
static auto up = "";
static auto down = "";

/// Outputs the raw thorin IR as a graph without performing any scope or scheduling analysis
struct DotPrinter {
    DotPrinter(World& world, const char* filename = "world.dot") : world(world) {
        file = std::ofstream(filename);
    }

private:
    void dump_def(const Def* def);
    void dump_literal(const Literal*);
    void dump_continuation(const Continuation* cont);

public:
    void print() {
        file << "digraph " << world.name() << " {" << up;
        file << endl << "bgcolor=transparent;";
        for (auto& external: world.externals()) {
            dump_def(external.second);
        }
        file << down << endl << "}" << endl;
    }

    bool print_lower_order_args = false;
    bool print_instanced_filters = false;

private:
    thorin::World& world;

    DefSet done;
    std::ofstream file;
};

void DotPrinter::dump_def(const Def* def) {
    if (done.contains(def))
        return;

    if (auto cont = def->isa_nom<Continuation>())
        dump_continuation(cont);
    else if (def->isa<Literal>())
        dump_literal(def->as<Literal>());
    else {
        // dump_def_generic(def, "red", "star");

        // default (primops)
        std::string color = "darkseagreen1";
        std::string style = "filled";
        std::string shape = "oval";
        std::string name = def->op_name();

        std::unique_ptr<std::vector<std::tuple<std::string, const Def*>>> filtered_ops;

        if (def->isa<Param>()) {
            color = "grey";
            shape = "oval";
        } else if (auto app = def->isa<App>()) {
            color = "darkgreen";

            filtered_ops = std::make_unique<std::vector<std::tuple<std::string, const Def*>>>();

            if (auto callee_cont = app->callee()->isa_nom<Continuation>()) {
                switch (callee_cont->intrinsic()) {
                    case Intrinsic::Branch: {
                        color = "lightblue";
                        name = "branch";
                        if (print_lower_order_args) {
                            (*filtered_ops).emplace_back("mem", app->arg(0));
                            (*filtered_ops).emplace_back("condition", app->arg(1));
                        }
                        (*filtered_ops).emplace_back("true", app->arg(2));
                        (*filtered_ops).emplace_back("false", app->arg(3));

                        goto print_node;
                    }
                    case Intrinsic::Match: {
                        color = "lightblue";
                        name = "match";

                        if (print_lower_order_args) {
                            (*filtered_ops).emplace_back("mem", app->arg(0));
                            (*filtered_ops).emplace_back("inspectee", app->arg(1));
                        }

                        (*filtered_ops).emplace_back("default_case", app->arg(2));
                        for (size_t i = 3; i < app->num_args(); i++) {
                            if (print_lower_order_args)
                                (*filtered_ops).emplace_back("case", app->arg(i));
                            else {
                                (*filtered_ops).emplace_back("case", app->arg(i)->as<Tuple>()->op(1));
                            }
                        }

                        goto print_node;
                    }
                    default: break;
                }
            }

            if (print_instanced_filters)
                (*filtered_ops).emplace_back("filter", app->filter());
            (*filtered_ops).emplace_back("callee", app->callee());
            for (size_t i = 0; i < app->num_args(); i++) {
                if (print_lower_order_args || app->arg(i)->type()->order() >= 1)
                    (*filtered_ops).emplace_back("arg"+std::to_string(i), app->arg(i));
            }
        } else if (auto variant_ctor = def->isa<Variant>()) {
            name = "variant(" + std::to_string(variant_ctor->index()) + ")";
        } else if (auto variant_extract = def->isa<VariantExtract>()) {
            name = "variant_extract(" + std::to_string(variant_extract->index()) + ")";
        }

        print_node:

        file << endl << def->unique_name() << " [" << up;

        file << endl << "label = \"";
        file << name;
        file << "\";";

        file << endl << "shape = " << shape << ";";
        file << endl << "style = " << style << ";";
        file << endl << "color = " << color << ";";

        file << down << endl << "]";

        done.emplace(def);

        if (!filtered_ops) {
            for (size_t i = 0; i < def->num_ops(); i++) {
                const auto& op = def->op(i);
                dump_def(op);
                file << endl << def->unique_name() << " -> " << op->unique_name() << " [arrowhead=vee,label=\"o" << i << "\",fontsize=8,fontcolor=grey];";
            }
        } else {
            for (auto [label, op] : *filtered_ops) {
                dump_def(op);
                file << endl << def->unique_name() << " -> " << op->unique_name() << " [arrowhead=vee,label=\"" << label << "\",fontsize=8,fontcolor=grey];";
            }
        }
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

void DotPrinter::dump_continuation(const Continuation* cont) {
    done.emplace(cont);
    auto intrinsic = cont->intrinsic();
    file << endl << cont->unique_name() << " [" << up;

    file << endl << "label = \"";
    if (cont->is_external())
        file << "[extern]\\n";
    auto name = cont->name();
    if (!cont->is_external())
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
    if (cont->is_external()) {
        file << endl << "color = pink;";
        file << endl << "style = filled;";
    }

    file << down << endl << "]";

    /*for (size_t i = 0; i < cont->num_args(); i++) {
        auto arg = cont->arg(i);
        dump_def(arg);

        if (cont->callee()->uses().size() > 1)
            file << endl << arg->unique_name() << " -> " << cont->callee()->unique_name() << " [arrowhead=onormal,label=\"a" << i << " from " << cont->unique_name() << "\",fontsize=8,fontcolor=grey];";
        else
            file << endl << arg->unique_name() << " -> " << cont->callee()->unique_name() << " [arrowhead=onormal,label=\"a" << i << "\",fontsize=8,fontcolor=grey];";
    }*/

    switch (intrinsic) {
        // We don't care about the params for these, or the callee
        case Intrinsic::Match:
        case Intrinsic::Branch:
            return;
        default:
            break;
    }

    /*for (size_t i = 0; i < cont->num_params(); i++) {
        auto param = cont->param(i);
        dump_def(param);
        file << endl << param->unique_name() << " -> " << cont->unique_name() << " [arrowhead=none,label=\"p" << i << "\",fontsize=8,fontcolor=grey];";
    }*/
    if (cont->has_body()) {
        dump_def(cont->body());
        file << endl << cont->unique_name() << " -> " << cont->body()->unique_name() << " [arrowhead=normal];";
    }
}

DEBUG_UTIL void dump_dot_world(World& world) {
    DotPrinter(world).print();
}

}

#endif //DOT_DUMP_H
