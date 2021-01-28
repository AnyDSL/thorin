#include "thorin/transform/mangle.h"
#include "thorin/world.h"
#include "thorin/util/log.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/domtree.h"

#include <iostream>
#include <map>

namespace thorin {

class Vectorizer {
public:
    Vectorizer(World &world)
        : world_(world)
        , boundary_(Def::gid_counter())
    {}
    bool run();

private:
    class DivergenceAnalysis {
    public:
        enum State {
            Varying,
            Uniform
        };

        Continuation *base;

        GIDMap<Continuation*, ContinuationSet> dominatedBy;
        GIDMap<Continuation*, ContinuationSet> reachableBy;
        GIDMap<Continuation*, ContinuationSet> loopLatches;
        GIDMap<Continuation*, ContinuationSet> loopExits;
        GIDMap<Continuation*, ContinuationSet> loopBodys;
        GIDMap<Continuation*, ContinuationSet> relJoins;

        GIDMap<const Def*, State> uniform;

        void computeLoops();

        ContinuationSet successors(Continuation *cont);
        ContinuationSet predecessors(Continuation *cont);

        DivergenceAnalysis(Continuation* base) : base(base) {};
        void run();
        State getUniform(const Def * def);

        friend class Vectorizer;
    };

    World& world_;
    size_t boundary_;
    ContinuationSet done_;
    ContinuationMap<bool> top_level_;
    Def2Def def2def_;
    DivergenceAnalysis * div_analysis_;

    size_t vector_width = 8;

    std::queue<Continuation*> queue_;
    void enqueue(Continuation* continuation) {
        if (continuation->gid() < 2 * boundary_ && done_.emplace(continuation).second)
            queue_.push(continuation);
    }

    const Type *widen(const Type *);
    const Def *widen(const Def *);
    Continuation *widen();

    void widen_setup(Continuation *);
    Continuation *kernel;
    bool widen_within(const Def *);

    void widen_body(Continuation *, Continuation *);
    Continuation* widen_head(Continuation* old_continuation);
};

Vectorizer::DivergenceAnalysis::State
Vectorizer::DivergenceAnalysis::getUniform(const Def * def) {
    return uniform[def];
}

ContinuationSet Vectorizer::DivergenceAnalysis::successors(Continuation * cont) {
    if (loopExits.contains(cont)) {
        return loopExits[cont];
    } else {
        ContinuationSet continues;
        for (auto node : cont->succs())
            if (!node->is_intrinsic())
                continues.emplace(node);
        return continues;
    }
}

ContinuationSet Vectorizer::DivergenceAnalysis::predecessors(Continuation * cont) {
    ContinuationSet nodes;
    for (auto pre : cont->preds())
        if (!pre->is_intrinsic())
            nodes.emplace(pre);
    for (auto it : loopExits) {
        if (nodes.contains(it.first))
            nodes.erase(it.first);
    }
    for (auto it : loopExits) {
        if (it.second.contains(cont))
            nodes.emplace(it.first);
    }
    return nodes;
}

void Vectorizer::DivergenceAnalysis::computeLoops() {
    std::queue <Continuation*> queue;
    ContinuationSet done;
    //Step 1: construct head rewired CFG. (should probably be implicit)
    //Step 1.1: Find Loop Headers

    //Step 1.1.1 Reachability analysis
    queue.push(base);

    while (!queue.empty()) {
        Continuation *cont = pop(queue);
        bool changed = false;

        if (!reachableBy.contains(cont)) {
            changed = true;
            reachableBy[cont] = ContinuationSet({cont});
        }
        ContinuationSet mydom = reachableBy[cont];

        ContinuationSet other;
        bool first = true;

        for (auto pre : cont->preds()) {
            if (!reachableBy.contains(pre)) {
                other.clear();
                break;
            }

            if (first) {
                first = false;
                other = reachableBy[pre];
                continue;
            }

            for (auto elem : reachableBy[pre]) {
                other.emplace(elem);
            }
        }

        for (auto elem : other) {
            changed = mydom.emplace(elem).second ? true : changed;
        }

        if (changed) {
            reachableBy[cont] = mydom;
            for (auto succ : cont->succs()) {
                if (!succ->is_intrinsic())
                    queue.push(succ);
            }
        }
    }

    dominatedBy = reachableBy;

    //Step 1.1.2: Dominance analysis from reachability analysis
    queue.push(base);

    while (!queue.empty()) {
        Continuation *cont = pop(queue);
        bool changed = false;

        assert (dominatedBy.contains(cont));
        ContinuationSet mydom = dominatedBy[cont];

        ContinuationSet other;
        bool first = true;

        for (auto pre : cont->preds()) {
            if (!dominatedBy.contains(pre)) {
                other.clear();
                break;
            }

            if (first) {
                first = false;
                other = dominatedBy[pre];
                continue;
            }

            ContinuationSet toDelete;

            for (auto elem : other) {
                if (!dominatedBy[pre].contains(elem))
                    toDelete.emplace(elem);
            }

            for (auto elem : toDelete) {
                other.erase(elem);
            }
        }

        ContinuationSet toDelete;

        for (auto elem : mydom) {
            if (!other.contains(elem))
                toDelete.emplace(elem);
        }
        toDelete.erase(cont);

        changed |= !toDelete.empty();

        for (auto elem : toDelete) {
            mydom.erase(elem);
        }

        if (changed || done.emplace(cont).second) {
            dominatedBy[cont] = mydom;
            for (auto succ : cont->succs()) {
                if (!succ->is_intrinsic())
                    queue.push(succ);
            }
        }
    }

#ifdef DUMP_LOOP_ANALYSIS
    base->dump();
    for (auto elem : reachableBy) {
        std::cerr << "reachable by\n";
        elem.first->dump();
        for (auto elem2 : elem.second)
            elem2->dump();
        std::cerr << "end\n";
    }

    for (auto elem : dominatedBy) {
        std::cerr << "dominated by\n";
        elem.first->dump();
        for (auto elem2 : elem.second)
            elem2->dump();
        std::cerr << "end\n";
    }
    std::cerr << "\n";
#endif

    done.clear();

    //Step 1.1.3: Find Loop Headers and Latches
    queue.push(base);
    done.emplace(base);

    while (!queue.empty()) {
        Continuation *cont = pop(queue);

#ifdef DUMP_LOOP_ANALYSIS
        std::cerr << "\n";
        cont->dump();
#endif

        auto mydom = dominatedBy[cont];

        for (auto succ : cont->succs()) {
            if (succ->is_intrinsic())
                continue;
#ifdef DUMP_LOOP_ANALYSIS
            succ->dump();
#endif
            if (mydom.contains(succ)) {
#ifdef DUMP_LOOP_ANALYSIS
                std::cerr << "Loop registered\n";
#endif
                loopLatches[succ].emplace(cont);
            }
            if (done.emplace(succ).second)
                queue.push(succ);
        }
    }

    for (auto it : loopLatches) {
        auto header = it.first;
        auto latches = it.second;

        for (auto latch : latches) {
            ContinuationSet reaching = reachableBy[latch];

            ContinuationSet toDelete;

            for (auto elem : reaching) {
                if (!dominatedBy[elem].contains(header))
                    toDelete.emplace(elem);
            }

            for (auto elem : toDelete)
                reaching.erase(elem);

            for (auto elem : reaching)
                loopBodys[header].emplace(elem);
        }
    }

    for (auto it : loopBodys) {
        auto header = it.first;
        auto body = it.second;

        for (auto elem : body)
            for (auto succ : elem->succs())
                if (!succ->is_intrinsic() && !body.contains(succ))
                    loopExits[header].emplace(succ);
    }
}

void Vectorizer::DivergenceAnalysis::run() {
    computeLoops();

#ifdef DUMP_DIV_ANALYSIS
    base->dump();
    std::cerr << "Loops are\n";
    for (auto elem : loopLatches) {
        std::cerr << "Header\n";
        elem.first->dump();
        std::cerr << "Latches\n";
        for (auto latch : elem.second)
            latch->dump();
        std::cerr << "Body\n";
        for (auto elem : loopBodys[elem.first])
            elem->dump();
        std::cerr << "Exits\n";
        for (auto elem : loopExits[elem.first])
            elem->dump();
    }
    std::cerr << "End Loops\n";
#endif

    std::queue <Continuation*> queue;
    ContinuationSet done;

    //Step 2: Find diverging paths in CFG.
    //Step 2.1: Find split nodes
    //Possible split nodes are nodes with multiple successors.
    ContinuationSet splitNodes;

    queue.push(base);
    done.emplace(base);

    while (!queue.empty()) {
        Continuation *cont = pop(queue);

        if (cont->succs().size() > 1) {
            splitNodes.emplace(cont);
#ifdef DUMP_DIV_ANALYSIS
            cont->dump();
#endif
        }

        for (auto succ : cont->succs())
            if (done.emplace(succ).second)
                queue.push(succ);
    }

#ifdef DUMP_DIV_ANALYSIS
    std::cerr << "Chapter 5\n";
#endif

    //Step 2.2: Chapter 5, alg. 1: Construct labelmaps.
    for (auto *split : splitNodes) {
        done.clear();

        GIDMap<Continuation*, Continuation*> LabelMap;

#ifdef DUMP_DIV_ANALYSIS
        std::cerr << "\nSplit analysis\n";
        split->dump();
#endif

        ContinuationSet Joins;

        for (auto succ : split->succs()) {
            if (succ->is_intrinsic())
                continue;
            queue.push(succ);
            LabelMap[succ] = succ;
        }

        while (!queue.empty()) {
            Continuation *cont = pop(queue);

            auto keys = predecessors(cont);
#ifdef DUMP_DIV_ANALYSIS
            std::cerr << "Predecessors\n";
            cont->dump();
            keys.dump();
#endif

            ContinuationSet toDelete;
            for (auto key : keys) {
                if (!LabelMap.contains(key))
                    toDelete.emplace(key);
            }
            for (auto key : toDelete)
                keys.erase(key);

            Continuation *oldkey = LabelMap[cont];

            if (!keys.empty()) {
                ContinuationSet plabs;
                for (auto key : keys)
                    if (LabelMap.contains(key))
                        plabs.emplace(LabelMap[key]);

                //At this point, we need to distinguish direct successors of split from the rest.
                //We know that nodes that are already labeled with themselves should not be updated,
                //but should be put into the set of relevant joins instead.

                Continuation* oldlabel = LabelMap[cont];
                if (oldlabel == cont) {
                    //This node was either already marked as a join or, more importantly, it was a start node.
                    if (plabs.size() > 1 || (plabs.size() == 1 && *plabs.begin() != cont))
                        Joins.emplace(cont);
                } else {
                    if (plabs.size() == 1)
                        LabelMap[cont] = *plabs.begin();
                    else {
                        LabelMap[cont] = cont;
                        Joins.emplace(cont);
                    }
                }
            }

            if (done.emplace(cont).second || oldkey != LabelMap[cont]) {
#ifdef DUMP_DIV_ANALYSIS
                std::cerr << "Successors\n";
                cont->dump();
                successors(cont).dump();
#endif
                for (auto succ : successors(cont))
                    queue.push(succ);
            }
        }

        relJoins[split] = Joins;

#ifdef DUMP_DIV_ANALYSIS
        std::cerr << "Split node\n";
        split->dump();
        std::cerr << "Labelmap:\n";
        LabelMap.dump();
        std::cerr << "Joins:\n";
        Joins.dump();
        std::cerr << "End\n";
#endif
    }

    //TODO: Reaching Definitions Analysis.
    //Step 3: Definite Reaching Definitions Analysis (see Chapter 6) (not at first, focus on CFG analysis first)
    //Note: I am currently not sure which is better, first doing the Definite Reachign Analysis, or first doing the CFG analysis.

    //Step 4: Vaule Uniformity
    //Step 4.1: Mark all Values as being uniform.
    Scope scope(base);
    for (auto def : scope.defs())
        uniform[def] = Uniform;

    std::queue <const Def*> def_queue;

    //Step 4.2: Mark varying defs
    //TODO: Memory Analysis: We need to track values that lie in memory slots!
    //This might not be so significant, stuff resides in memory for a reason.

    //Step 4.2.1: Mark incomming trip counter as varying.
    //for (auto def : base->params_as_defs()) {
    //    uniform[def] = Varying;
    //    def_queue.push(def);
    //}
    auto def = base->params_as_defs()[1];
    uniform[def] = Varying;
    def_queue.push(def);

    //Step 4.2.2: Mark everything in relevant joins as varying (for now at least)
    for (auto it : relJoins) {
        Continuation *split = it.first;
        //use split to capture information about the branching condition
        const Continuation * branch_int = split->op(0)->isa_continuation();
        const Def * branch_cond;
        if (branch_int && (branch_int->intrinsic() == Intrinsic::Branch || branch_int->intrinsic() == Intrinsic::Match))
            branch_cond = split->op(1);
        if (branch_cond && uniform[branch_cond] == Uniform)
            continue;
        ContinuationSet joins = it.second;
        for (auto join : joins) {
            Scope scope(join);
#ifdef DUMP_DIV_ANALYSIS
            std::cerr << "Varying values in\n";
            scope.dump();
#endif
            for (auto def : scope.defs()) {
                uniform[def] = Varying;
                def_queue.push(def);
            }
        }
    }

#ifdef DUMP_DIV_ANALYSIS
    for (auto it : relJoins) {
        it.first->dump();
        it.second.dump();
    }

    std::cerr << "\n";
#endif

    //Step 4.3: Profit?
    while (!def_queue.empty()) {
        const Def *def = pop(def_queue);
#ifdef DUMP_DIV_ANALYSIS
        std::cerr << "Will analyze ";
        def->dump();
#endif

        if (uniform[def] == Uniform)
            continue;

        for (auto use : def->uses()) {
            auto old_state = uniform[use];

            if (old_state == Uniform) {
                //Communicate Uniformity over continuation parameters
                Continuation *cont = use.def()->isa_continuation();
                if (cont) {
#ifdef DUMP_DIV_ANALYSIS
                    cont->dump();
#endif
                    bool is_op = false; //TODO: this is not a good filter for finding continuation calls!
                    int opnum = 0;
                    for (auto param : cont->ops()) {
                        if (param == def) {
                            is_op = true;
                            break;
                        }
                        opnum++;
                    }
#ifdef DUMP_DIV_ANALYSIS
                    cont->dump();
                    std::cerr << is_op << "\n";
                    std::cerr << opnum << "\n";
#endif
                    auto target = cont->ops()[0]->isa_continuation();
                    if (is_op && target && target->is_intrinsic() && opnum == 1 && relJoins.find(cont) != relJoins.end()) {
                        ContinuationSet joins = relJoins[cont];
                        for (auto join : joins) {
                            Scope scope(join);
#ifdef DUMP_DIV_ANALYSIS
                            std::cerr << "Varying values in\n";
                            scope.dump();
#endif
                            for (auto def : scope.defs()) { //TODO: only parameters are varying, not the entire continuation!
#ifdef DUMP_DIV_ANALYSIS
                                std::cerr << "Push def ";
                                def->dump();
#endif
                                uniform[def] = Varying;
                                def_queue.push(def);
                            }
                        }
                    } else if (target && is_op) {
                        for (size_t i = 1; i < cont->num_ops(); ++i) {
                            auto source_param = cont->op(i);
                            auto target_param = target->params_as_defs()[i - 1];

                            if (!uniform.contains(source_param))
                                continue;

                            if (uniform[source_param] == Varying && uniform[target_param] == Uniform) {
                                uniform[target_param] = Varying;
                                def_queue.push(target_param);
#ifdef DUMP_DIV_ANALYSIS
                                std::cerr << "Push param ";
                                target_param->dump();
#endif
                            }
                        }
#ifdef DUMP_DIV_ANALYSIS
                    } else {
                        std::cerr << "\nNot Found\n";
                        cont->dump();
                        for (auto it : relJoins) {
                            it.first->dump();
                            it.second.dump();
                        }
                        std::cerr << "\n";
#endif
                    }
                } else {
                    uniform[use] = Varying;
                    def_queue.push(use);
                    //std::cerr << "Push ";
                    //use->dump();
                }
            }
        }
    }

#ifdef DUMP_DIV_ANALYSIS
    std::cerr << "\n";

    for (auto uni : uniform) {
        if (uni.second == Varying)
            std::cerr << "Varying ";
        if (uni.second == Uniform)
            std::cerr << "Uniform ";
        uni.first->dump();
    }
#endif

}

const Type *Vectorizer::widen(const Type *old_type) {
    return world_.vec_type(old_type, vector_width);
}

Continuation* Vectorizer::widen_head(Continuation* old_continuation) {
    assert(!def2def_.contains(old_continuation));
    assert(!old_continuation->empty());
    Continuation* new_continuation;

    std::vector<const Type*> param_types;
    for (size_t i = 0, e = old_continuation->num_params(); i != e; ++i) {
        if (!is_mem(old_continuation->param(i)) && div_analysis_->getUniform(old_continuation->param(i)) != DivergenceAnalysis::State::Uniform)
            param_types.emplace_back(widen(old_continuation->param(i)->type()));
        else
            param_types.emplace_back(old_continuation->param(i)->type());
    }

    auto fn_type = world_.fn_type(param_types);
    new_continuation = world_.continuation(fn_type, old_continuation->debug_history());

    def2def_[old_continuation] = new_continuation;

    for (size_t i = 0, e = old_continuation->num_params(); i != e; ++i)
        def2def_[old_continuation->param(i)] = new_continuation->param(i);

    return new_continuation;
}

const Def* Vectorizer::widen(const Def* old_def) {
    if (auto new_def = find(def2def_, old_def)) {
        return new_def;
    } else if (!widen_within(old_def)) {
        if (auto cont = old_def->isa_continuation()) {
            if (cont->is_intrinsic() && cont->intrinsic() == Intrinsic::Match) {
                auto type = old_def->type();
                auto match = world_.match(world_.vec_type(type->op(0), vector_width), type->num_ops() - 2);
                return def2def_[old_def] = match;
            }
            if (cont->is_intrinsic() && cont->intrinsic() == Intrinsic::Branch) {
                auto branch = world_.continuation(world_.fn_type({world_.vec_type(world_.type_bool(), vector_width), world_.fn_type(), world_.fn_type()}), Intrinsic::Branch, {"br_vec"});
                return def2def_[old_def] = branch;
            }
        }
        return old_def;
    } else if (auto old_continuation = old_def->isa_continuation()) {
        auto new_continuation = widen_head(old_continuation);
        widen_body(old_continuation, new_continuation);
        return new_continuation;
    } else if (div_analysis_->getUniform(old_def) == DivergenceAnalysis::State::Uniform) {
        return old_def; //TODO: this def could contain a continuation inside a tuple for match cases!
    } else if (auto param = old_def->isa<Param>()) {
        widen(param->continuation());
        assert(def2def_.contains(param));
        return def2def_[param];
    } else if (auto param = old_def->isa<Extract>()) {
        Array<const Def*> nops(param->num_ops());

        nops[0] = widen(param->op(0));
        if (nops[0]->type()->isa<VectorExtendedType>())
            nops[1] = world_.tuple({world_.top(param->op(1)->type()), param->op(1)});
        else
            nops[1] = param->op(1);

        auto type = widen(param->type());
        const Def* new_primop;
        if (param->isa<PrimLit>()) {
            assert(false); // This should not be reachable!
            Array<const Def*> elements(vector_width);
            for (size_t i = 0; i < vector_width; i++) {
                elements[i] = param;
            }
            new_primop = world_.vector(elements, param->debug_history());
        } else {
            new_primop = param->rebuild(nops, type);
        }
        return def2def_[param] = new_primop;
    } else if (auto param = old_def->isa<VariantExtract>()) {
        Array<const Def*> nops(param->num_ops());

        nops[0] = widen(param->op(0));

        auto type = widen(param->type());
        const Def* new_primop;
        if (param->isa<PrimLit>()) {
            assert(false); // This should not be reachable!
            Array<const Def*> elements(vector_width);
            for (size_t i = 0; i < vector_width; i++) {
                elements[i] = param;
            }
            new_primop = world_.vector(elements, param->debug_history());
        } else {
            new_primop = param->rebuild(nops, type);
        }
        return def2def_[param] = new_primop;
    } else if (auto param = old_def->isa<ArithOp>()) {
        Array<const Def*> nops(param->num_ops());
        bool any_vector = false;
        for (size_t i = 0, e = param->num_ops(); i != e; ++i) {
            nops[i] = widen(param->op(i));
            if (auto vector = nops[i]->type()->isa<VectorType>())
                any_vector |= vector->is_vector();
        }

        if (any_vector) {
            for (size_t i = 0, e = param->num_ops(); i != e; ++i) {
                if (auto vector = nops[i]->type()->isa<VectorType>())
                    if (vector->is_vector())
                        continue;

                //non-vector element in a vector setting needs to be extended to a vector.
                Array<const Def*> elements(vector_width);
                for (size_t j = 0; j < vector_width; j++) {
                    elements[j] = nops[i];
                }
                nops[i] = world_.vector(elements, nops[i]->debug_history());
            }
        }

        auto type = widen(param->type());
        const Def* new_primop;
        if (param->isa<PrimLit>()) {
            Array<const Def*> elements(vector_width);
            for (size_t i = 0; i < vector_width; i++) {
                elements[i] = param;
            }
            new_primop = world_.vector(elements, param->debug_history());
        } else {
            new_primop = param->rebuild(nops, type);
        }
        return def2def_[param] = new_primop;
    } else {
        auto old_primop = old_def->as<PrimOp>();
        Array<const Def*> nops(old_primop->num_ops());
        bool any_vector = false;
        for (size_t i = 0, e = old_primop->num_ops(); i != e; ++i) {
            nops[i] = widen(old_primop->op(i));
            if (nops[i]->type()->isa<VectorExtendedType>())
                any_vector = true;
        }

        if (any_vector && (old_primop->isa<BinOp>() || old_primop->isa<Access>())) {
            for (size_t i = 0, e = old_primop->num_ops(); any_vector && i != e; ++i) {
                if (nops[i]->type()->isa<VectorExtendedType>())
                    continue;
                if (nops[i]->type()->isa<MemType>())
                    continue;
                Array<const Def*> elements(vector_width);
                for (size_t j = 0; j < vector_width; j++)
                    elements[j] = nops[i];
                nops[i] = world_.vector(elements, nops[i]->debug_history());
            }
        }

        auto type = widen(old_primop->type());
        const Def* new_primop;

        if (old_primop->isa<PrimLit>()) {
            assert(false && "Primlits are uniform");
        } else {
            new_primop = old_primop->rebuild(nops, type);
        }
        return def2def_[old_primop] = new_primop;
    }
}

void Vectorizer::widen_body(Continuation* old_continuation, Continuation* new_continuation) {
    assert(!old_continuation->empty());

    // fold branch and match
    if (auto callee = old_continuation->callee()->isa_continuation()) {
        switch (callee->intrinsic()) {
            case Intrinsic::Branch: {
                if (auto lit = widen(old_continuation->arg(0))->isa<PrimLit>()) {
                    auto cont = lit->value().get_bool() ? old_continuation->arg(1) : old_continuation->arg(2);
                    return new_continuation->jump(widen(cont), {}, old_continuation->jump_debug());
                }
                break;
            }
            case Intrinsic::Match:
                if (old_continuation->num_args() == 2)
                    return new_continuation->jump(widen(old_continuation->arg(1)), {}, old_continuation->jump_debug());

                if (auto lit = widen(old_continuation->arg(0))->isa<PrimLit>()) {
                    for (size_t i = 2; i < old_continuation->num_args(); i++) {
                        auto new_arg = widen(old_continuation->arg(i));
                        if (world_.extract(new_arg, 0_s)->as<PrimLit>() == lit)
                            return new_continuation->jump(world_.extract(new_arg, 1), {}, old_continuation->jump_debug());
                    }
                }
                break;
            default:
                break;
        }
    }

    Array<const Def*> nops(old_continuation->num_ops());
    for (size_t i = 0, e = nops.size(); i != e; ++i)
        nops[i] = widen(old_continuation->op(i));

    Array<const Def*> nargs(nops.size() - 1); //new args of new_continuation
    const Def* ntarget = nops.front();   // new target of new_continuation

    if (auto callee = old_continuation->callee()->isa_continuation()) {
        if (callee->intrinsic() == Intrinsic::Branch || callee->intrinsic() == Intrinsic::Match) {
            if (!nops[1]->type()->isa<VectorExtendedType>()) {
                ntarget = old_continuation->op(0);
            }
        }
        if (callee->is_imported()) {
            auto old_fn_type = callee->type()->as<FnType>();
            Array<const Type*> ops(old_fn_type->num_ops());
            for (size_t i = 0; i < old_fn_type->num_ops(); i++)
                ops[i] = nops[i + 1]->type(); //TODO: this feels like a bad hack. At least it's working for now.

            Debug de = callee->debug();
            if (de.name() == "llvm.exp.f32")
                de.set("llvm.exp.v8f32"); //TODO: Use vectorlength to find the correct intrinsic.
            else if (de.name() == "llvm.exp.f64")
                de.set("llvm.exp.v8f64");
            else {
                std::cerr << "Not supported: " << de.name() << "\n";
                assert(false && "Import not supported in vectorize.");
            }

            ntarget = world_.continuation(world_.fn_type(ops), callee->attributes(), de);
            def2def_[callee] = ntarget;
        }
    }

    Scope scope(kernel);

    if (old_continuation->op(0)->isa<Continuation>() && scope.contains(old_continuation->op(0))) {
        auto oldtarget = old_continuation->op(0)->as<Continuation>();

        for (size_t i = 0; i < nops.size() - 1; i++) {
            auto arg = nops[i + 1];
            if (!is_mem(arg) &&
                    !arg->type()->isa<VectorExtendedType>() && //TODO: This is not correct.
                    div_analysis_->getUniform(oldtarget->param(i)) != DivergenceAnalysis::State::Uniform) {
                Array<const Def*> elements(vector_width);
                for (size_t i = 0; i < vector_width; i++) {
                    elements[i] = arg;
                }

                auto new_param = world_.vector(elements, arg->debug_history());
                nargs[i] = new_param;
            } else {
                nargs[i] = arg;
            }
        }
    } else {
        for (size_t i = 0; i < nops.size() - 1; i++)
            nargs[i] = nops[i + 1];
    }

    for (size_t i = 0, e = nops.size() - 1; i != e; ++i) {
            //if (div_analysis_->getUniform(old_continuation->op(i)) == DivergenceAnalysis::State::Uniform &&
            //        div_analysis_->getUniform(oldtarget->param(i-1)) != DivergenceAnalysis::State::Uniform) {
            if (!nargs[i]->type()->isa<VectorExtendedType>() &&
                    ntarget->isa<Continuation>() &&
                    ntarget->as<Continuation>()->param(i)->type()->isa<VectorExtendedType>()) { //TODO: base this on divergence analysis
                Array<const Def*> elements(vector_width);
                for (size_t j = 0; j < vector_width; j++)
                    elements[j] = nargs[i];
                nargs[i] = world_.vector(elements, nargs[i]->debug_history());
            }
    }

    new_continuation->jump(ntarget, nargs, old_continuation->jump_debug());
}

Continuation *Vectorizer::widen() {
    Scope scope(kernel);
    auto defs = Defs();

    Continuation *ncontinuation;

    // create new_entry - but first collect and specialize all param types
    std::vector<const Type*> param_types;
    for (size_t i = 0, e = scope.entry()->num_params(); i != e; ++i) {
        if (i == 1) {
            param_types.emplace_back(widen(scope.entry()->param(i)->type()));
        } else {
            param_types.emplace_back(scope.entry()->param(i)->type());
        }
    }

    auto fn_type = world_.fn_type(param_types);
    ncontinuation = world_.continuation(fn_type, scope.entry()->debug_history());

    // map value params
    def2def_[scope.entry()] = scope.entry();
    for (size_t i = 0, j = 0, e = scope.entry()->num_params(); i != e; ++i) {
        auto old_param = scope.entry()->param(i);
        auto new_param = ncontinuation->param(j++);
        def2def_[old_param] = new_param;
        new_param->debug().set(old_param->name());
    }

    for (auto def : defs)
        def2def_[def] = ncontinuation->append_param(def->type());

    // mangle filter
    if (!scope.entry()->filter().empty()) {
        Array<const Def*> new_filter(ncontinuation->num_params());
        size_t j = 0;
        for (size_t i = 0, e = scope.entry()->num_params(); i != e; ++i) {
            new_filter[j++] = widen(scope.entry()->filter(i));
        }

        for (size_t e = ncontinuation->num_params(); j != e; ++j)
            new_filter[j] = world_.literal_bool(false, Debug{});

        ncontinuation->set_filter(new_filter);
    }

    widen_body(scope.entry(), ncontinuation);

    return ncontinuation;
}

void Vectorizer::widen_setup(Continuation* kern) {
    kernel = kern;
}

bool Vectorizer::widen_within(const Def* def) {
    Scope scope(kernel);
    return scope.contains(def);
}

bool Vectorizer::run() {
    world_.dump();

    for (auto continuation : world_.exported_continuations()) {
        enqueue(continuation);
        top_level_[continuation] = true;
    }

    //Task 1: Divergence Analysis
    //Task 1.1: Find all vectorization continuations
    while (!queue_.empty()) {
        Continuation *cont = pop(queue_);

        if (cont->intrinsic() == Intrinsic::Vectorize) {
            std::cerr << "Continuation\n";
            //cont->dump_head();
            //std::cerr << "\n";

            for (auto pred : cont->preds()) {
                auto *kernarg = dynamic_cast<const Global *>(pred->arg(2));
                assert(kernarg && "Not sure if all kernels are always declared globally");
                assert(!kernarg->is_mutable() && "Self transforming code is not supported here!");
                auto *kerndef = kernarg->init()->isa_continuation();
                assert(kerndef && "We need a continuation for vectorization");

    //Task 1.2: Divergence Analysis for each vectorize block
    //Warning: Will fail to produce meaningful results or rightout break the program if kerndef does not dominate its subprogram
                div_analysis_ = new DivergenceAnalysis(kerndef);
                div_analysis_->run();

    //Task 2: Widening
                widen_setup(kerndef);
                auto *vectorized = widen();
                //auto *vectorized = clone(Scope(kerndef));
                def2def_[kerndef] = vectorized;

    //Task 3: Linearize divergent controll flow
                for (auto it : div_analysis_->relJoins) {
                    auto latch_old = it.first;
                    Continuation * latch = const_cast<Continuation*>(def2def_[latch_old]->as<Continuation>());
                    assert(latch);

                    assert (it.second.size() <= 1 && "no complex controll flow");
                    auto join_old = *it.second.begin();
                    Continuation *join;
                    if(join_old == *it.second.end()) {
                        //Only possible option is that all cases call return directly.
                        const FnType* join_type = world_.fn_type({world_.mem_type()});
                        join = world_.continuation(join_type, Debug(Symbol("join")));
                        auto return_intrinsic = vectorized->param(2);
                        //TODO: only replace calls that are dominated by latch.
                        for (auto& use : return_intrinsic->copy_uses()) {
                            auto def = const_cast<Def*>(use.def());
                            auto index = use.index();
                            def->unset_op(index);
                            def->set_op(index, join);
                        }

                        join->jump(return_intrinsic, {join->param(0)});
                    } else {
                       join = const_cast<Continuation*>(def2def_[join_old]->as<Continuation>());
                    }
                    assert(join);

                    //cases according to latch: match and branch.
                    //match:
                    auto cont = latch->op(0)->isa_continuation();
                    if (cont && cont->is_intrinsic() && cont->intrinsic() == Intrinsic::Match && latch->arg(0)->type()->isa<VectorExtendedType>()) {
                        Scope latch_scope(latch);
                        Schedule latch_schedule = schedule(latch_scope);

                        auto vec_mask = world_.vec_type(world_.type_bool(), vector_width);
                        auto variant_index = latch->arg(0);

                        for (size_t i = 1; i < latch_old->num_args(); i++) {
                            const Def * old_case = latch_old->arg(i);
                            if (i != 1) {
                                assert(old_case->isa<Tuple>());
                                old_case = old_case->as<Tuple>()->op(1);
                            }
                            Continuation * new_case = const_cast<Continuation*>(def2def_[old_case]->as<Continuation>());
                            assert(new_case);

                            auto new_mem = new_case->append_param(world_.mem_type());
                            Def* first_mem = nullptr;
                            for (auto &block : latch_schedule) {
                                if (block.continuation() == new_case) {
                                    for (auto primop : block) {
                                        if (auto memop = primop->isa<MemOp>()) {
                                            first_mem = const_cast<MemOp*>(memop);
                                            break;
                                        }
                                    }
                                    break;
                                }
                            }
                            if (first_mem) {
                                int index = 0; //TODO: always correct?
                                first_mem->unset_op(index);
                                first_mem->set_op(index, new_mem);
                            } else {
                                new_case->update_arg(0, new_mem);
                            }
                        }

                        const Def *mem = latch->mem_param();
                        const Schedule::Block *latch_block = latch_schedule.begin();
                        for (auto primop : *latch_block) {
                            if (auto memop = primop->isa<MemOp>())
                                mem = memop->out_mem();
                        }

                        Array<const Def *> join_cache(join->num_params() - 1);
                        for (size_t i = 0; i < join->num_params() - 1; i++) {
                            auto t = world_.alloc(join->param(i + 1)->type(), mem);
                            mem = world_.extract(t, (int) 0);
                            join_cache[i] = world_.extract(t, 1);
                        }

                        Array<const Def*> predicates(latch->num_args() - 1);
                        for (size_t i = 1; i < latch->num_args() - 1; i++) {
                            auto elem = latch->arg(i + 1);
                            auto val = elem->as<Tuple>()->op(0)->as<PrimLit>();
                            Array<const Def *> elements(vector_width);
                            for (size_t i = 0; i < vector_width; i++) {
                                elements[i] = val;
                            }
                            auto val_vec = world_.vector(elements);
                            auto pred = world_.cmp(Cmp_eq, variant_index, val_vec);
                            predicates[i] = pred;
                        }
                        predicates[0] = predicates[1];
                        for (size_t i = 2; i < latch->num_args() - 1; i++) {
                            predicates[0] = world_.binop(ArithOp_or, predicates[0], predicates[i]);
                        }
                        predicates[0] = world_.arithop_not(predicates[0]);

                        const Def * otherwise_old = latch_old->arg(1);
                        Continuation * otherwise = const_cast<Continuation*>(def2def_[otherwise_old]->as<Continuation>());
                        assert(otherwise);

                        Continuation * current_case = otherwise;
                        Continuation * case_back = world_.continuation(world_.fn_type({world_.mem_type()}), Debug(Symbol("otherwise_back")));

                        auto new_jump = world_.predicated(vec_mask);
                        latch->jump(new_jump, { mem, predicates[0], current_case, case_back }, latch->jump_location());

                        Continuation * pre_join = world_.continuation(world_.fn_type({world_.mem_type()}), Debug(Symbol("match_merge")));

                        for (size_t i = 2; i < latch_old->num_args() + 1; i++) {
                            Scope case_scope(current_case);

                            Continuation *next_case;
                            Continuation *next_case_back;

                            if (i < latch_old->num_args()) {
                                auto next_case_old = latch_old->arg(i)->as<Tuple>()->op(1);
                                next_case = const_cast<Continuation*>(def2def_[next_case_old]->as<Continuation>());
                                next_case_back = world_.continuation(world_.fn_type({world_.mem_type()}), Debug(Symbol("case_back")));
                            } else {
                                next_case = pre_join;
                                next_case_back = pre_join;
                            }

                            assert(next_case);
                            assert(next_case_back);

                            for (auto pred : join->preds()) {
                                if (!case_scope.contains(pred))
                                    continue;

                                assert(pred->arg(0)->type()->isa<MemType>());
                                mem = pred->arg(0);

                                for (size_t j = 1; j < pred->num_args(); j++) {
                                    mem = world_.store(mem, join_cache[j - 1], pred->arg(j));
                                }

                                current_case->jump(case_back, { mem });

                                auto new_jump = world_.predicated(vec_mask);
                                const Def* predicate;
                                if (i < latch_old->num_args())
                                    predicate = predicates[i - 1];
                                else {
                                    auto true_elem = world_.one(world_.type_bool());
                                    Array<const Def *> elements(vector_width);
                                    for (size_t i = 0; i < vector_width; i++) {
                                        elements[i] = true_elem;
                                    }
                                    predicate = world_.vector(elements);
                                }
                                case_back->jump(new_jump, { case_back->mem_param(), predicate, next_case, next_case_back }, latch->jump_location());
                            }

                            current_case = next_case;
                            case_back = next_case_back;
                        }

                        mem = pre_join->mem_param();
                        Array<const Def*> join_params(join->num_params());
                        for (size_t i = 1; i < join->num_params(); i++) {
                            auto load = world_.load(mem, join_cache[i - 1]);
                            auto value = world_.extract(load, (int) 1);
                            mem = world_.extract(load, (int) 0);
                            join_params[i] = value;
                        }
                        join_params[0] = mem;
                        pre_join->jump(join, join_params, latch->jump_location());
                    }

                    if (cont && cont->is_intrinsic() && cont->intrinsic() == Intrinsic::Branch && latch->arg(0)->type()->isa<VectorExtendedType>()) {
                        const Def * then_old = latch_old->arg(1);
                        Continuation * then_new = const_cast<Continuation*>(def2def_[then_old]->as<Continuation>());
                        assert(then_new);

                        const Def * else_old = latch_old->arg(2);
                        Continuation * else_new = const_cast<Continuation*>(def2def_[else_old]->as<Continuation>());
                        assert(else_new);

                        auto vec_mask = world_.vec_type(world_.type_bool(), vector_width);

                        Scope latch_scope(latch);
                        const Def *mem = latch->mem_param();
                        Schedule latch_schedule = schedule(latch_scope);
                        const Schedule::Block *latch_block = latch_schedule.begin();
                        for (auto primop : *latch_block) {
                            if (auto memop = primop->isa<MemOp>())
                                mem = memop->out_mem();
                        }

                        Array<const Def *> join_cache(then_new->num_args() - 1);

                        for (size_t i = 0; i < then_new->num_args() - 1; i++) {
                            auto t = world_.alloc(then_new->arg(i + 1)->type(), mem);
                            mem = world_.extract(t, (int) 0);
                            join_cache[i] = world_.extract(t, 1);
                        }

                        const Def* predicate_true = latch->arg(0);
                        const Def* predicate_false = world_.arithop_not(predicate_true);

                        auto then_mem = then_new->append_param(world_.mem_type());
                        Def* then_first_mem = nullptr;
                        for (auto &block : latch_schedule) {
                            if (block.continuation() == then_new) {
                                for (auto primop : block) {
                                    if (auto memop = primop->isa<MemOp>()) {
                                        then_first_mem = const_cast<MemOp*>(memop);
                                        break;
                                    }
                                }
                                break;
                            }
                        }
                        if (then_first_mem) {
                            int index = 0;
                            then_first_mem->unset_op(index);
                            then_first_mem->set_op(index, then_mem);
                        } else {
                            then_new->update_arg(0, then_mem);
                        }

                        auto else_mem = else_new->append_param(world_.mem_type());
                        Def* else_first_mem = nullptr;
                        for (auto &block : latch_schedule) {
                            if (block.continuation() == else_new) {
                                for (auto primop : block) {
                                    if (auto memop = primop->isa<MemOp>()) {
                                        else_first_mem = const_cast<MemOp*>(memop);
                                        break;
                                    }
                                }
                                break;
                            }
                        }
                        if (else_first_mem) {
                            int index = 0;
                            else_first_mem->unset_op(index);
                            else_first_mem->set_op(index, else_mem);
                        } else {
                            else_new->update_arg(0, else_mem);
                        }

                        Continuation * pre_join = world_.continuation(world_.fn_type({world_.mem_type()}), Debug(Symbol("branch_merge")));
                        Continuation * then_new_back = world_.continuation(world_.fn_type({world_.mem_type()}), Debug(Symbol("branch_true_back")));
                        Continuation * else_new_back = world_.continuation(world_.fn_type({world_.mem_type()}), Debug(Symbol("branch_false_back")));

                        Scope then_scope(then_new);

                        auto new_jump = world_.predicated(vec_mask);
                        latch->jump(new_jump, { mem, predicate_true, then_new, then_new_back }, latch->jump_location());

                        for (auto pred : join->preds()) {
                            if (!then_scope.contains(pred))
                                continue;

                            //get old mem parameter, if possible.
                            assert(pred->arg(0)->type()->isa<MemType>());
                            mem = pred->arg(0);

                            for (size_t j = 1; j < pred->num_args(); j++) {
                                mem = world_.store(mem, join_cache[j - 1], pred->arg(j));
                            }

                            pred->jump(then_new_back, { mem });

                            auto new_jump = world_.predicated(vec_mask);
                            then_new_back->jump(new_jump, { then_new_back->mem_param(), predicate_false, else_new, else_new_back }, latch->jump_location());
                        }

                        Scope else_scope(else_new);

                        for (auto pred : join->preds()) {
                            if (!else_scope.contains(pred))
                                continue;

                            auto true_elem = world_.one(world_.type_bool());
                            Array<const Def *> elements(vector_width);
                            for (size_t i = 0; i < vector_width; i++) {
                                elements[i] = true_elem;
                            }
                            auto one_predicate = world_.vector(elements);

                            assert(pred->arg(0)->type()->isa<MemType>());
                            mem = pred->arg(0);

                            for (size_t j = 1; j < pred->num_args(); j++) {
                                mem = world_.store(mem, join_cache[j - 1], pred->arg(j));
                            }

                            pred->jump(else_new_back, {mem});

                            auto new_jump = world_.predicated(vec_mask);
                            else_new_back->jump(new_jump, { else_new_back->mem_param(), one_predicate, pre_join, pre_join }, latch->jump_location());
                        }

                        mem = pre_join->param(0);
                        Array<const Def*> join_params(join->num_params());
                        for (size_t i = 1; i < join->num_params(); i++) {
                            auto load = world_.load(mem, join_cache[i - 1]);
                            auto value = world_.extract(load, (int) 1);
                            mem = world_.extract(load, (int) 0);
                            join_params[i] = value;
                        }
                        join_params[0] = mem;
                        pre_join->jump(join, join_params, latch->jump_location());
                    }
                }

                delete div_analysis_;

    //Task 4: Rewrite vectorize call
                if (vectorized) {
                    for (auto caller : cont->preds()) {
                        Array<const Def*> args(vectorized->num_params());

                        args[0] = caller->arg(0); //mem
                        //args[1] = caller->arg(1); //width
                        Array<const Def*> defs(vector_width);
                        for (size_t i = 0; i < vector_width; i++) {
                            defs[i] = world_.literal_qs32(i, caller->arg(1)->debug_history());
                        }
                        args[1] = world_.vector(defs, caller->arg(1)->debug_history());

                        for (size_t p = 2; p < vectorized->num_params(); p++) {
                            args[p] = caller->arg(p + 1);
                        }

                        caller->jump(vectorized, args, caller->jump_debug());
                    }
                }
            }
            std::cerr << "Continuation end\n\n";
        }

        for (auto succ : cont->succs())
            enqueue(succ);
    }

    world_.cleanup();
    world_.dump();

    return false;
}

bool vectorize(World& world) {
    VLOG("start vectorizer");
    auto res = Vectorizer(world).run();
    VLOG("end vectorizer");
    return res;
}

}
