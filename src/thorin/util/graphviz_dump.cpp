#ifndef DOT_DUMP_H
#define DOT_DUMP_H

#include "thorin/world.h"
#include "thorin/analyses/scope.h"

namespace thorin {

/// Outputs the raw thorin IR as a graph without performing any scope or scheduling analysis
struct DotPrinter {
    DotPrinter(World& world, const char* filename = "world.dot") : world_(world), forest_(world) {
        file = std::ofstream(filename);
        begin();
    }

    ~DotPrinter() {
        end();
    }
private:
    int indent = 0;
    std::string endl() {
        std::string s = "\n";
        for (int i = 0; i < indent; i++)
            s += " ";
        return s;
    }
    std::string up() {
        indent++;
        return "";
    }
    std::string down() {
        indent--;
        return "";
    }

    void begin() {
        file << "digraph " << "world" << " {" << up();
        file << endl() << "bgcolor=transparent;";
    }

    void end() {
        file << arrows.str();
        file << down() << endl() << "}" << endl();
    }

    std::string def_id(const Def* def) {
        return std::string(tag2str(def->tag())) + "_" + std::to_string(def->gid());
    }

    std::string emit_def(const Def* def);
    std::string dump_literal(const Literal*);
    std::string dump_continuation(Continuation* cont);

    std::stringstream arrows;

    void arrow(const std::string& src, const std::string& dst, const std::string& extra) {
        if (src.empty() || dst.empty())
            return;
        arrows << endl() << src << " -> " << dst << " " << extra << ";";
    }
public:
    std::string dump_def(const Def* def);

    void run() {
        delay_printing_ops = false;
        while (!todo.empty()) {
            auto def = todo.pop();
            dump_def(def);
        }
    }

    void print_scope(Scope& scope) {
        file << "subgraph cluster_" << u_++ << " {" << up() << endl();

        auto cont = scope.entry();
        emit_def(cont);

        for (auto child : scope.children_scopes())
            print_scope(scope.forest().get_scope(child));

        for (auto def : scope.defs()) {
            if (!done.contains(def) && !def->isa<Continuation>()) {
                emit_def(def);
            }
        }

        file << down() << endl() << "}" << endl();
    };

    bool print_lower_order_args = true;
    bool print_instanced_filters = false;
    bool print_literals = true;
    bool delay_printing_ops = true;
    Scope* single_scope = nullptr;

private:
    thorin::World& world_;
    ScopesForest forest_;

    int u_ = 0;

    unique_queue<DefSet> todo;
    DefMap<std::string> done;
    std::ofstream file;
};

std::string DotPrinter::dump_def(const Def* def) {
    if (done.contains(def))
        return done[def];

    if (delay_printing_ops) {
        todo.push(def);
        return def_id(def);
    }

    return emit_def(def);
}

std::string DotPrinter::emit_def(const Def* def) {
    assert(!done.contains(def));
    if (auto cont = def->isa_nom<Continuation>())
        return dump_continuation(cont);
    else if (def->isa<Literal>())
        return dump_literal(def->as<Literal>());
    else {
        // default (primops)
        std::string color = "";
        std::string fillcolor = "darkseagreen1";
        std::string style = "filled";
        std::string shape = "oval";
        std::string label = std::string(def->op_name()) + " :: " + def->name();

        std::unique_ptr<std::vector<std::tuple<std::string, const Def*>>> filtered_ops;

        if (single_scope && single_scope->free_frontier().contains(def))
            color = "blue";

        if (def->isa<Param>()) {
            fillcolor = "grey";
            shape = "oval";
            label = def->unique_name();
        } else if (auto app = def->isa<App>()) {
            fillcolor = "darkgreen";

            filtered_ops = std::make_unique<std::vector<std::tuple<std::string, const Def*>>>();

            if (auto callee_cont = app->callee()->isa_nom<Continuation>()) {
                switch (callee_cont->intrinsic()) {
                    case Intrinsic::Branch: {
                        fillcolor = "lightblue";
                        label = "branch";
                        if (print_lower_order_args) {
                            (*filtered_ops).emplace_back("mem", app->arg(0));
                            (*filtered_ops).emplace_back("condition", app->arg(1));
                        }
                        (*filtered_ops).emplace_back("true", app->arg(2));
                        (*filtered_ops).emplace_back("false", app->arg(3));

                        goto print_node;
                    }
                    case Intrinsic::Match: {
                        fillcolor = "lightblue";
                        label = "match";

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
            label = "variant(" + std::to_string(variant_ctor->index()) + ")";
        } else if (auto variant_extract = def->isa<VariantExtract>()) {
            label = "variant_extract(" + std::to_string(variant_extract->index()) + ")";
        }

        print_node:

        file << endl() << def_id(def) << " [" << up();

        file << endl() << "label = \"";
        file << label;
        file << "\";";

        file << endl() << "shape = " << shape << ";";
        file << endl() << "style = " << style << ";";
        file << endl() << "fillcolor = " << fillcolor << ";";
        if (color != "")
            file << endl() << "color = " << color << ";";

        file << down() << endl() << "]";
        done.emplace(def, def_id(def));

        if (!filtered_ops) {
            for (size_t i = 0; i < def->num_ops(); i++) {
                const auto& op = def->op(i);
                arrow(def_id(def), dump_def(op), "[arrowhead=vee,label=\"o" + std::to_string(i) + "\",fontsize=8,fontcolor=grey]");
            }
        } else {
            for (auto [edge_label, op] : *filtered_ops) {
                arrow(def_id(def), dump_def(op), "[arrowhead=vee,label=\"" + edge_label + "\",fontsize=8,fontcolor=grey]");
            }
        }

        return def_id(def);
    }
}

std::string DotPrinter::dump_literal(const Literal* def) {
    if (!print_literals)
        return "";
    assert(def->num_ops() == 0);
    file << endl() << def_id(def) << " [" << up();

    file << endl() << "label = \"";
    file << def->to_string();
    file << "\";";

    file << endl() << "style = dotted;";

    file << down() << endl() << "]";

    done.emplace(def, def_id(def));
    return def_id(def);
}

std::string DotPrinter::dump_continuation(Continuation* cont) {
    done.emplace(cont, def_id(cont));
    auto intrinsic = cont->intrinsic();
    file << endl() << def_id(cont) << " [" << up();

    file << endl() << "label = \"";
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

    file << endl() << "shape = rectangle;";
    if (intrinsic != Intrinsic::None) {
        file << endl() << "color = lightblue;";
        file << endl() << "style = filled;";
    }
    if (cont->is_external()) {
        file << endl() << "color = pink;";
        file << endl() << "style = filled;";
    }

    file << down() << endl() << "]";

    if (cont->has_body())
        arrow(def_id(cont), dump_def(cont->body()), "[arrowhead=normal]");

    return def_id(cont);
}

DEBUG_UTIL void dump_dot_world(World& world) {
    DotPrinter printer(world);
    for (auto& external: world.externals()) {
        printer.dump_def(external.second);
    }
    printer.run();
}

DEBUG_UTIL void dump_dot_def(const Def* def) {
    DotPrinter printer(def->world());
    printer.dump_def(def);
    printer.run();
}

DEBUG_UTIL void dump_dot_scopes(World& world) {
    DotPrinter printer(world);
    ScopesForest forest(world);
    for (auto& top_level : forest.top_level_scopes()) {
        auto& scope = forest.get_scope(top_level);
        printer.print_scope(scope);
    }
    printer.run();
}

DEBUG_UTIL void dump_dot_scope(Scope& scope) {
    DotPrinter printer(scope.world());
    printer.single_scope = &scope;
    printer.print_scope(scope);
    printer.run();
}

}

#endif //DOT_DUMP_H
