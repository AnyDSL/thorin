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
    std::string dump_def(const Def* def);
    std::string dump_literal(const Literal*);
    std::string dump_continuation(Continuation* cont);

    void arrow(const std::string& src, const std::string& dst, const std::string& extra) {
        if (src.empty() || dst.empty())
            return;
        file << endl << src << " -> " << dst << " " << extra << ";";
    }

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
    bool print_literals = false;

private:
    thorin::World& world;

    DefMap<std::string> done;
    std::ofstream file;
};

std::string DotPrinter::dump_def(const Def* def) {
    if (done.contains(def))
        return done[def];

    if (auto cont = def->isa_nom<Continuation>())
        return dump_continuation(cont);
    else if (def->isa<Literal>())
        return dump_literal(def->as<Literal>());
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
        done.emplace(def, def->unique_name());

        if (!filtered_ops) {
            for (size_t i = 0; i < def->num_ops(); i++) {
                const auto& op = def->op(i);
                arrow(def->unique_name(), dump_def(op), "[arrowhead=vee,label=\"o" + std::to_string(i) + "\",fontsize=8,fontcolor=grey]");
            }
        } else {
            for (auto [label, op] : *filtered_ops) {
                arrow(def->unique_name(), dump_def(op), "[arrowhead=vee,label=\"" + label + "\",fontsize=8,fontcolor=grey]");
            }
        }

        return def->unique_name();
    }
}

std::string DotPrinter::dump_literal(const Literal* def) {
    if (!print_literals)
        return "";
    assert(def->num_ops() == 0);
    file << endl << def->unique_name() << " [" << up;

    file << endl << "label = \"";
    file << def->to_string();
    file << "\";";

    file << endl << "style = dotted;";

    file << down << endl << "]";

    done.emplace(def, def->unique_name());
    return def->unique_name();
}

std::string DotPrinter::dump_continuation(Continuation* cont) {
    done.emplace(cont, cont->unique_name());
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

    if (cont->has_body())
        arrow(cont->unique_name(), dump_def(cont->body()), "[arrowhead=normal]");

    return cont->unique_name();
}

DEBUG_UTIL void dump_dot_world(World& world) {
    DotPrinter(world).print();
}

}

#endif //DOT_DUMP_H
