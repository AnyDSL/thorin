#ifndef DOT_DUMP_H
#define DOT_DUMP_H

#include "thorin/world.h"
#include "thorin/analyses/scope.h"

namespace thorin {

static auto endl = "\n";
static auto up = "";
static auto down = "";

/// Outputs the raw thorin IR as a graph without performing any scope or scheduling analysis
struct DotPrinter {
    DotPrinter(World& world, const char* filename = "world.dot") : world_(world), forest_(world) {
        file = std::ofstream(filename);
    }

private:
    std::string def_id(const Def* def) {
        return std::string(tag2str(def->tag())) + "_" + std::to_string(def->gid());
    }

    std::string emit_def(const Def* def);
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
        if (print_scopes) {
            schedule();
        }

        file << "digraph " << "world" << " {" << up;
        file << endl << "bgcolor=transparent;";

        if (print_scopes) {
            for (auto cont : world_.copy_continuations()) {
                //if (!forest_.get_scope(cont).has_free_params())
                if (forest_.get_scope(cont).parent_scope() == nullptr)
                    visit_scope(cont);
            }

            for (auto def : world_.defs()) {
                if (!done.contains(def)) {
                    emit_def(def);
                }
            }
        } else {
            for (auto& external: world_.externals()) {
                emit_def(external.second);
            }
        }

        file << down << endl << "}" << endl;
    }

    bool print_lower_order_args = false;
    bool print_instanced_filters = false;
    bool print_literals = true;
    bool print_scopes = true;

private:
    void visit_scope(Continuation* cont) {
        file << "subgraph cluster_" << u_++ << " {" << up << endl;

        emit_def(cont);

        for (auto child : nesting_[cont])
            visit_scope(child);

        Scope& scope = forest_.get_scope(cont);
        for (auto def : scope.defs()) {
            if (!done.contains(def) && def != cont) {
                emit_def(def);
            }
        }

        file << down << endl << "}" << endl;
    };

    void schedule() {
        // build forest of continuations
        for (auto cont : world_.copy_continuations()) {
            auto parent = forest_.get_scope(cont).parent_scope();
            if (parent)
                nesting_[parent].push_back(cont);
        }
    }

    thorin::World& world_;
    ScopesForest forest_;

    int u_ = 0;

    ContinuationMap<std::vector<Continuation*>> nesting_;
    DefMap<Continuation*> scheduling_;

    DefMap<std::string> done;
    std::ofstream file;
};

std::string DotPrinter::dump_def(const Def* def) {
    if (done.contains(def))
        return done[def];

    if (print_scopes) {
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

        file << endl << def_id(def) << " [" << up;

        file << endl << "label = \"";
        file << name;
        file << "\";";

        file << endl << "shape = " << shape << ";";
        file << endl << "style = " << style << ";";
        file << endl << "color = " << color << ";";

        file << down << endl << "]";
        done.emplace(def, def_id(def));

        if (!filtered_ops) {
            for (size_t i = 0; i < def->num_ops(); i++) {
                const auto& op = def->op(i);
                arrow(def_id(def), dump_def(op), "[arrowhead=vee,label=\"o" + std::to_string(i) + "\",fontsize=8,fontcolor=grey]");
            }
        } else {
            for (auto [label, op] : *filtered_ops) {
                arrow(def_id(def), dump_def(op), "[arrowhead=vee,label=\"" + label + "\",fontsize=8,fontcolor=grey]");
            }
        }

        return def_id(def);
    }
}

std::string DotPrinter::dump_literal(const Literal* def) {
    if (!print_literals)
        return "";
    assert(def->num_ops() == 0);
    file << endl << def_id(def) << " [" << up;

    file << endl << "label = \"";
    file << def->to_string();
    file << "\";";

    file << endl << "style = dotted;";

    file << down << endl << "]";

    done.emplace(def, def_id(def));
    return def_id(def);
}

std::string DotPrinter::dump_continuation(Continuation* cont) {
    done.emplace(cont, def_id(cont));
    auto intrinsic = cont->intrinsic();
    file << endl << def_id(cont) << " [" << up;

    file << endl << "label = \"";
    if (cont->is_external())
        file << "[extern]\\n";
    auto name = cont->name();
    if (!cont->is_external())
        name = def_id(cont);

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
        arrow(def_id(cont), dump_def(cont->body()), "[arrowhead=normal]");

    //auto& scope = forest_.get_scope(cont);
    //auto parent = scope.parent_scope();
    //if (parent)
    //    arrow(def_id(cont), dump_def(parent), "[arrowhead=normal,color=teal]");

    return def_id(cont);
}

DEBUG_UTIL void dump_dot_world(World& world) {
    DotPrinter(world).print();
}

}

#endif //DOT_DUMP_H
