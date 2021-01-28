#include "thorin/transform/mangle.h"
#include "thorin/world.h"
#include "thorin/util/log.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/domtree.h"

#include <llvm/Support/Timer.h>
#include <llvm/Support/raw_ostream.h>

#include <iostream>
#include <map>

//#define DUMP_DIV_ANALYSIS
//#define DUMP_WIDEN
//#define DUMP_VECTORIZER

namespace thorin {

class Vectorizer {
public:
    Vectorizer(World &world)
        : world_(world)
        , boundary_(Def::gid_counter())
        , time("Vec", "Vectorize")
        , time_div("Div", "Divergence Analysis")
        , time_widen("Widen", "Widen")
        , time_clean("Cleanup", "Cleanup")
        , time_lin("Linearization", "Linearization")
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
        GIDMap<Continuation*, ContinuationSet> splitParrents;

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

    llvm::Timer time;
    llvm::Timer time_div;
    llvm::Timer time_widen;
    llvm::Timer time_clean;
    llvm::Timer time_lin;

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

    void Continuation_MemParam(Continuation* continuation, Schedule& vector_schedule, Schedule::Map<const Def*>& block2mem);
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
            if (cont->callee()->isa<Continuation>() && cont->callee()->as<Continuation>()->is_intrinsic()) {
                splitNodes.emplace(cont);
#ifdef DUMP_DIV_ANALYSIS
                cont->dump();
            } else {
                std::cerr << "Multiple successors in non-intrinsic node\n";
                cont->dump();
#endif
            }
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
                    plabs.emplace(LabelMap[key]);

#ifdef DUMP_DIV_ANALYSIS
                std::cerr << "Previous\n";
                cont->dump();
                plabs.dump();
#endif

                //At this point, we need to distinguish direct successors of split from the rest.
                //We know that nodes that are already labeled with themselves should not be updated,
                //but should be put into the set of relevant joins instead.

                Continuation* oldlabel = LabelMap[cont];
                if (oldlabel == cont) {
                    //This node was either already marked as a join or, more importantly, it was a start node.
                    if (predecessors(cont).contains(split)) {
                        if ((plabs.size() > 1) || (plabs.size() == 1 && *plabs.begin() != cont))
                            Joins.emplace(cont);
                    } else {
                        if (plabs.size() == 1 && *plabs.begin() != cont) {
                            LabelMap[cont] = *plabs.begin();
                            Joins.erase(cont);
                        }
                    }
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

    //Step 2.3: Sort Split-Joins in a hirarchy
    for (auto it : relJoins) {
        for (auto it_inner : relJoins) {
            if (it_inner.first == it.first)
                continue;
            if (!Scope(it.first).contains(it_inner.first))
                continue;
            bool contained = it_inner.second.size() > 0;
            for (auto join : it_inner.second)
                if (!it.second.contains(join)) {
                    contained = false;
                    break;
                }
            if (contained)
                //it_inner.first is fully contained in a branch of it.first
                splitParrents[it_inner.first].emplace(it.first);
        }
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
        //Continuation *split = it.first;
        //use split to capture information about the branching condition
        /*const Continuation * branch_int = split->op(0)->isa_continuation();
        const Def * branch_cond;
        if (branch_int && (branch_int->intrinsic() == Intrinsic::Branch || branch_int->intrinsic() == Intrinsic::Match))
            branch_cond = split->op(1);
        if (branch_cond && uniform[branch_cond] == Uniform)
            continue;*/ //TODO: in this case, we still need to communicate varying information across parameters.
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
#ifdef DUMP_DIV_ANALYSIS
                    std::cerr << "Push ";
                    use->dump();
#endif
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
#ifdef DUMP_WIDEN
    std::cout << "Widen\n";
    old_def->dump();
#endif

    if (auto new_def = find(def2def_, old_def)) {
#ifdef DUMP_WIDEN
        std::cout << "Found\n";
#endif
        return new_def;
    } else if (!widen_within(old_def)) {
#ifdef DUMP_WIDEN
        std::cout << "NWithin\n";
#endif
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
#ifdef DUMP_WIDEN
        std::cout << "Contin\n";
#endif
        auto new_continuation = widen_head(old_continuation);
        widen_body(old_continuation, new_continuation);
        return new_continuation;
    } else if (div_analysis_->getUniform(old_def) == DivergenceAnalysis::State::Uniform) {
#ifdef DUMP_WIDEN
        std::cout << "Uni\n";
#endif
        return old_def; //TODO: this def could contain a continuation inside a tuple for match cases!
    } else if (auto param = old_def->isa<Param>()) {
#ifdef DUMP_WIDEN
        std::cout << "Param\n";
#endif
        widen(param->continuation());
        assert(def2def_.contains(param));
        return def2def_[param];
    } else if (auto param = old_def->isa<Extract>()) {
#ifdef DUMP_WIDEN
        std::cout << "Extract\n";
#endif
        Array<const Def*> nops(param->num_ops());

        nops[0] = widen(param->op(0));
        if (nops[0]->type()->isa<VectorExtendedType>())
            nops[1] = world_.tuple({world_.top(param->op(1)->type()), param->op(1)});
        else
            nops[1] = param->op(1);

        auto type = widen(param->type());
        const Def* new_primop;
        if (param->isa<PrimLit>()) {
            assert(false && "This should not be reachable!");
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
#ifdef DUMP_WIDEN
        std::cout << "VarExtract\n";
#endif
        Array<const Def*> nops(param->num_ops());

        nops[0] = widen(param->op(0));

        auto type = widen(param->type());
        const Def* new_primop;
        if (param->isa<PrimLit>()) {
            assert(false && "This should not be reachable!");
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
#ifdef DUMP_WIDEN
        std::cout << "Arith\n";
#endif
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
#ifdef DUMP_WIDEN
        std::cout << "Primop\n";
#endif
        auto old_primop = old_def->as<PrimOp>();
        Array<const Def*> nops(old_primop->num_ops());
        bool any_vector = false;
        for (size_t i = 0, e = old_primop->num_ops(); i != e; ++i) {
            nops[i] = widen(old_primop->op(i));
            if (nops[i]->type()->isa<VectorExtendedType>())
                any_vector = true;
        }

        if (any_vector && (old_primop->isa<BinOp>() || old_primop->isa<Access>())) {
            for (size_t i = 0, e = old_primop->num_ops(); i != e; ++i) {
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

        const Type* type;
        if (any_vector)
          type  = widen(old_primop->type());
        else
          type  = old_primop->type();

        const Def* new_primop;

        if (old_primop->isa<PrimLit>()) {
            assert(false && "Primlits are uniform");
        } else {
            new_primop = old_primop->rebuild(nops, type);
        }
        //assert(new_primop->type() == type);
        return def2def_[old_primop] = new_primop;
    }
}

void Vectorizer::widen_body(Continuation* old_continuation, Continuation* new_continuation) {
    assert(!old_continuation->empty());
#ifdef DUMP_WIDEN
    std::cout << "Body\n";
#endif

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
            else if (de.name() == "llvm.sqrt.f32")
                de.set("llvm.sqrt.v8f32");
            else if (de.name() == "llvm.sqrt.f64")
                de.set("llvm.sqrt.v8f64");
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
#ifdef DUMP_WIDEN
    std::cout << "Jump\n";
#endif
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


void Vectorizer::Continuation_MemParam(Continuation* continuation, Schedule &vector_schedule, Schedule::Map<const Def*>& block2mem) {
    auto continuation_mem = continuation->append_param(world_.mem_type());
    Def* continuation_first_mem = nullptr;
    const Schedule::Block* continuation_block;
    for (auto &block : vector_schedule) {
        if (block.continuation() == continuation) {
            continuation_block = &block;
            for (auto primop : block) {
                if (auto memop = primop->isa<MemOp>()) {
                    continuation_first_mem = const_cast<MemOp*>(memop);
                    break;
                }
            }
        }
    }
    assert(continuation_block);
    if (continuation_first_mem) {
        int index = 0;
        assert(is_mem(continuation_first_mem->op(index)));
        continuation_first_mem->unset_op(index);
        continuation_first_mem->set_op(index, continuation_mem);
    } else {
        if (is_mem(continuation->arg(0)))
            continuation->update_arg(0, continuation_mem);
        else {
            auto& outmem = block2mem[*continuation_block];
            for (auto use : outmem->uses()) {
                if (Scope(continuation).contains(use)) {
                    int index = use.index();
                    Def* olduse = const_cast<Def*>(use.def());
                    olduse->unset_op(index);
                    olduse->set_op(index, continuation_mem);
                }
            }
        }
    }
}


bool Vectorizer::run() {
#ifdef DUMP_VECTORIZER
    world_.dump();
#endif

    llvm::TimeRegion vregion(time);

    for (auto continuation : world_.exported_continuations()) {
        enqueue(continuation);
        top_level_[continuation] = true;
    }

    //Task 1: Divergence Analysis
    //Task 1.1: Find all vectorization continuations
    while (!queue_.empty()) {
        Continuation *cont = pop(queue_);

        if (cont->intrinsic() == Intrinsic::Vectorize) {
#ifdef DUMP_VECTORIZER
            std::cerr << "Continuation\n";
            cont->dump_head();
            std::cerr << "\n";
#endif

            for (auto pred : cont->preds()) {
                auto *kernarg = dynamic_cast<const Global *>(pred->arg(2));
                assert(kernarg && "Not sure if all kernels are always declared globally");
                assert(!kernarg->is_mutable() && "Self transforming code is not supported here!");
                auto *kerndef = kernarg->init()->isa_continuation();
                assert(kerndef && "We need a continuation for vectorization");

    //Task 1.2: Divergence Analysis for each vectorize block
    //Task 1.2.1: Ensure the return intrinsic is only called once, to make the job of the divergence-analysis easier.
                auto ret_param = kerndef->ret_param();
#ifdef DUMP_VECTORIZER
                ret_param->dump();
#endif
                if (ret_param->uses().size() > 1) {
                    auto ret_type = const_cast<FnType*>(ret_param->type()->as<FnType>());
                    Continuation * ret_join = world_.continuation(ret_type, Debug(Symbol("shim")));
                    ret_param->replace(ret_join);

                    Array<const Def*> args(ret_join->num_params());
                    for (size_t i = 0, e = ret_join->num_params(); i < e; i++) {
                        args[i] = ret_join->param(i);
                    }

                    ret_join->jump(ret_param, args);
#ifdef DUMP_VECTORIZER
                    Scope(kerndef).dump();
#endif
                }
    //Warning: Will fail to produce meaningful results or rightout break the program if kerndef does not dominate its subprogram
                {
                    llvm::TimeRegion div_time(time_div);
                    div_analysis_ = new DivergenceAnalysis(kerndef);
                    div_analysis_->run();
                }

    //Task 2: Widening
                Continuation* vectorized;
                {
                    llvm::TimeRegion widen_time(time_widen);
                    widen_setup(kerndef);
                    vectorized = widen();
                }
                //auto *vectorized = clone(Scope(kerndef));
                def2def_[kerndef] = vectorized;

#ifdef DUMP_VECTORIZER
                Scope(vectorized).dump();
#endif

                {
                llvm::TimeRegion lin_time(time_lin);

                std::queue <Continuation*> split_queue;
                for (auto it : div_analysis_->relJoins) {
                    split_queue.push(it.first);
                }

    //Task 3: Linearize divergent controll flow
                //for (auto it : div_analysis_->relJoins) {
                while (!split_queue.empty()) {
                    Continuation* latch_old = pop(split_queue);
                    ContinuationSet joins_old = div_analysis_->relJoins[latch_old];
                    if (div_analysis_->splitParrents.contains(latch_old)) {
                        //Dont want to work with those just yet
                        continue;
                    }
                    Continuation * latch = const_cast<Continuation*>(def2def_[latch_old]->as<Continuation>());
                    assert(latch);

#ifdef DUMP_VECTORIZER
                    Scope(latch).dump(); //The mem monad can be malformed here already!?

                    if (joins_old.size() > 1) {
                        Scope(latch_old).dump();
                        joins_old.dump();
                        Scope(latch).dump();
                    }
#endif

                    Continuation *join;
                    if (joins_old.size() == 0) { //TODO: I will end up here with some intrinsics that will be recognized as splits, for some reason.
                        assert(false && "This should no longer happen");
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
                    } else if (joins_old.size() == 1) {
                        auto join_old = *joins_old.begin();
                        join = const_cast<Continuation*>(def2def_[join_old]->as<Continuation>());
                        assert(join);
                    } else {
                        Continuation* join_old = nullptr;
                        //TODO: this might go horribly wrong when loops are taken into account!
                        for (Continuation* join_it : joins_old) {
#ifdef DUMP_VECTORIZER
                            join_it->dump();
#endif
                            if (!join_old || div_analysis_->reachableBy[join_it].contains(join_old)) {
                                join_old = join_it;
                            }
                        }
                        assert(join_old);
                        for (auto join_it : joins_old) {
                            assert(div_analysis_->reachableBy[join_old].contains(join_it));
                        }
                        join = const_cast<Continuation*>(def2def_[join_old]->as<Continuation>());
                        assert(join);
                    }

                    //cases according to latch: match and branch.
                    //match:
                    auto cont = latch->op(0)->isa_continuation();
                    if (cont && cont->is_intrinsic() && cont->intrinsic() == Intrinsic::Match && latch->arg(0)->type()->isa<VectorExtendedType>()) {
                        assert (joins_old.size() <= 1 && "no complex controll flow match");

                        //Scope latch_scope(latch);
                        //Schedule latch_schedule = schedule(latch_scope);
                        Scope vector_scope(vectorized);
                        Schedule vector_schedule = schedule(vector_scope);
                        auto& domtree = vector_schedule.cfg().domtree();
                        Schedule::Map<const Def*> block2mem(vector_schedule);
                        const Schedule::Block* latch_block;
                        for (auto& block : vector_schedule) {
                            const Def* mem = block.continuation()->mem_param();
                            auto idom = block.continuation() != vector_schedule.scope().entry() ? domtree.idom(block.node()) : block.node();
                            mem = mem ? mem : block2mem[(vector_schedule)[idom]];
                            for (auto primop : block) {
                                if (auto memop = primop->isa<MemOp>()) {
                                    mem = memop->out_mem();
                                }
                            }
                            block2mem[block] = mem;
                            if (block.continuation() == latch)
                                latch_block = &block;
                        }
                        assert(latch_block);
                        const Def *mem = block2mem[*latch_block];;
                        assert(mem);

                        auto vec_mask = world_.vec_type(world_.type_bool(), vector_width);
                        auto variant_index = latch->arg(0);

                        //Add memory parameters to make rewiring easier.
                        for (size_t i = 1; i < latch_old->num_args(); i++) {
                            const Def * old_case = latch_old->arg(i);
                            if (i != 1) {
                                assert(old_case->isa<Tuple>());
                                old_case = old_case->as<Tuple>()->op(1);
                            }
                            Continuation * new_case = const_cast<Continuation*>(def2def_[old_case]->as<Continuation>());
                            assert(new_case);
                            Continuation_MemParam(new_case, vector_schedule, block2mem);
                        }

                        //Allocate cache for overwritten objects.
                        size_t cache_size = join->num_params();
                        bool join_is_first_mem = false;
                        if (cache_size && is_mem(join->param(0))) {
                            cache_size -= 1;
                            join_is_first_mem = true;
                        }
                        assert(join_is_first_mem);

                        Array<const Def *> join_cache(cache_size);
                        for (size_t i = 0; i < cache_size; i++) {
                            auto t = world_.alloc(join->param(join_is_first_mem ? i + 1: i)->type(), mem);
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

                        auto new_jump_latch = world_.predicated(vec_mask);
                        latch->jump(new_jump_latch, { mem, predicates[0], current_case, case_back }, latch->jump_location());

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
                                    mem = world_.store(mem, join_cache[join_is_first_mem ? j - 1 : j], pred->arg(j));
                                }

                                current_case->jump(case_back, { mem });

                                auto new_jump_case = world_.predicated(vec_mask);
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
                                case_back->jump(new_jump_case, { case_back->mem_param(), predicate, next_case, next_case_back }, latch->jump_location());
                            }

                            current_case = next_case;
                            case_back = next_case_back;
                        }

                        mem = pre_join->mem_param();
                        Array<const Def*> join_params(join->num_params());
                        for (size_t i = 1; i < join->num_params(); i++) {
                            auto load = world_.load(mem, join_cache[join_is_first_mem ? i - 1 : i]);
                            auto value = world_.extract(load, (int) 1);
                            mem = world_.extract(load, (int) 0);
                            join_params[i] = value;
                        }
                        join_params[0] = mem;
                        pre_join->jump(join, join_params, latch->jump_location());
                    }

#ifdef DUMP_VECTORIZER
                    Scope(vectorized).dump();
#endif

                    if (cont && cont->is_intrinsic() && cont->intrinsic() == Intrinsic::Branch && latch->arg(0)->type()->isa<VectorExtendedType>()) {
                        //assert (joins_old.size() <= 1 && "no complex controll flow branch");
                        if (joins_old.size() > 1) {
                            //This should only occur when the condition contains an "and" or "or" instruction.
                            //TODO: try to assert this somehow?
                            assert(joins_old.size() == 2 && "Only this case is supported for now.");
                        }

                        const Def * then_old = latch_old->arg(1);
                        Continuation * then_new = const_cast<Continuation*>(def2def_[then_old]->as<Continuation>());
                        assert(then_new);

                        const Def * else_old = latch_old->arg(2);
                        Continuation * else_new = const_cast<Continuation*>(def2def_[else_old]->as<Continuation>());
                        assert(else_new);

#ifdef DUMP_VECTORIZER
                        std::cerr << "\n";
#endif

                        if (joins_old.size() > 1) {
                            assert(then_old->as<Continuation>()->succs().size() == 3 && "Additional branching should occur on the 'then' side.");
                            assert(else_old->as<Continuation>()->succs().size() == 1 && "Additional branching should occur on the 'then' side.");

#ifdef DUMP_VECTORIZER
                            for (auto elem : joins_old)
                                elem->dump();
                            for (auto elem : then_old->as<Continuation>()->succs())
                                elem->dump();
#endif

                            assert(then_old->as<Continuation>()->succs()[2]->succs().size() == 1);
                            assert(joins_old.contains(then_old->as<Continuation>()->succs()[2]->succs()[0]));
                        }

                        auto vec_mask = world_.vec_type(world_.type_bool(), vector_width);

#ifdef DUMP_VECTORIZER
                        Scope latch_scope(latch);

                        std::cerr << "Pre transformation\n";
                        Scope(latch_old).dump();
                        latch_scope.dump();
#endif

                        Scope vector_scope(vectorized);
                        Schedule vector_schedule = schedule(vector_scope);
                        auto& domtree = vector_schedule.cfg().domtree();
                        Schedule::Map<const Def*> block2mem(vector_schedule);
                        const Schedule::Block* latch_block;
                        for (auto& block : vector_schedule) {
                            const Def* mem = block.continuation()->mem_param();
                            auto idom = block.continuation() != vector_schedule.scope().entry() ? domtree.idom(block.node()) : block.node();
                            mem = mem ? mem : block2mem[(vector_schedule)[idom]];
                            for (auto primop : block) {
                                if (auto memop = primop->isa<MemOp>()) {
                                    mem = memop->out_mem();
                                }
                            }
                            block2mem[block] = mem;
                            if (block.continuation() == latch)
                                latch_block = &block;
                        }
                        assert(latch_block);
                        const Def *mem = block2mem[*latch_block];;
                        assert(mem);

                        //TODO: I might need to add mem to the br_vec intrinsic very early.

                        size_t cache_size = join->num_params();
                        bool join_is_first_mem = false;
                        if (cache_size && is_mem(join->param(0))) {
                            cache_size -= 1;
                            join_is_first_mem = true;
                        }
                        Array<const Def *> join_cache(cache_size);

                        for (size_t i = 0; i < cache_size; i++) {
                            auto t = world_.alloc(join->param(join_is_first_mem ? i + 1: i)->type(), mem);
                            mem = world_.extract(t, (int) 0);
                            join_cache[i] = world_.extract(t, 1);
                        }

                        const Def * pred_cache;
                        if (joins_old.size() > 1) {
                            auto false_elem = world_.zero(world_.type_bool());
                            Array<const Def *> elements(vector_width);
                            for (size_t i = 0; i < vector_width; i++) {
                                elements[i] = false_elem;
                            }
                            auto zero_predicate = world_.vector(elements);

                            auto t = world_.alloc(zero_predicate->type(), mem);
                            mem = world_.extract(t, (int) 0);
                            pred_cache = world_.extract(t, 1);

                            mem = world_.store(mem, pred_cache, zero_predicate);
                        }

                        const Def* predicate_true = latch->arg(0);
                        const Def* predicate_false = world_.arithop_not(predicate_true);

                        //TODO: This code still feels incomplete and broken. Devise a better way to handle this rewiring of the mem-monad.
                        //Maybe I should extend all cases surrounding a latch with a memory operand with code similar to schedule verification.
                        //I depend on the schedule from time to time. This is not good.

#ifdef DUMP_VECTORIZER
                        std::cerr << "PreAll\n";
                        Scope(latch).dump();
#endif

                        Continuation_MemParam(then_new, vector_schedule, block2mem);
                        Continuation_MemParam(else_new, vector_schedule, block2mem);

#ifdef DUMP_VECTORIZER
                        Scope(latch).dump();
#endif

                        Continuation * pre_join = world_.continuation(world_.fn_type({world_.mem_type()}), Debug(Symbol("branch_merge")));
                        Continuation * then_new_back = world_.continuation(world_.fn_type({world_.mem_type()}), Debug(Symbol("branch_true_back")));
                        Continuation * else_new_back = world_.continuation(world_.fn_type({world_.mem_type()}), Debug(Symbol("branch_false_back")));

                        Scope then_scope(then_new);

                        auto new_jump_latch = world_.predicated(vec_mask);
                        latch->jump(new_jump_latch, { mem, predicate_true, then_new, then_new_back }, latch->jump_location());

#ifdef DUMP_VECTORIZER
                        std::cerr << "Then\n";
#endif
                        //TODO: this might be an issue: There might be loops with the exit being the "then" case, then and else should be switched then!

                        for (auto join_old : joins_old) {
                            auto join = const_cast<Continuation*>(def2def_[join_old]->as<Continuation>());
                            assert(join);
                            for (auto pred : join->preds()) {
                                if (!then_scope.contains(pred))
                                    continue;

                                //get old mem parameter, if possible.
                                assert(pred->arg(0)->type()->isa<MemType>());
                                mem = pred->arg(0);

                                for (size_t j = 0; j < pred->num_args(); j++) {
                                    if (join_is_first_mem && j == 0)
                                        continue;
                                    mem = world_.store(mem, join_cache[join_is_first_mem ? j - 1 : j], pred->arg(j));
                                }

                                pred->jump(then_new_back, { mem });
                            }
                        }

                        auto new_jump_then = world_.predicated(vec_mask);
                        then_new_back->jump(new_jump_then, { then_new_back->mem_param(), predicate_false, else_new, else_new_back }, latch->jump_location());

#ifdef DUMP_VECTORIZER
                        Scope(latch).dump();
#endif
                        if (joins_old.size() > 1) { //TODO: There might be a structure of splits at the beginning of this block!
                            Continuation* next_target = const_cast<Continuation*>(then_new->arg(1)->as<Continuation>());
                            Continuation_MemParam(next_target, vector_schedule, block2mem);
                            Continuation* next_back = const_cast<Continuation*>(then_new->arg(2)->as<Continuation>());

#ifdef DUMP_VECTORIZER
                            next_back->dump();
#endif
                            assert(next_back->arg(0)->type()->isa<MemType>());
                            mem = next_back->arg(0);

                            Continuation_MemParam(next_back, vector_schedule, block2mem);

                            auto additional_predicate = then_new->arg(0);
                            if (!additional_predicate->type()->isa<VectorExtendedType>()) {
                                Array<const Def *> elements(vector_width);
                                for (size_t i = 0; i < vector_width; i++) {
                                    elements[i] = additional_predicate;
                                }
                                additional_predicate = world_.vector(elements);
                            }
                            auto new_predicate = world_.binop(ArithOp_and, predicate_true, additional_predicate);
                            auto new_jump = world_.predicated(vec_mask);

                            mem = world_.store(mem, pred_cache, new_predicate);

#ifdef DUMP_VECTORIZER
                            Scope(then_new).dump();
#endif
                            then_new->jump(new_jump, { mem, new_predicate, next_target, next_back }, latch->jump_location());
#ifdef DUMP_VECTORIZER
                            Scope(then_new).dump();
#endif
                        }

                        Scope else_scope(else_new);
#ifdef DUMP_VECTORIZER
                        Scope(latch).dump();
                        std::cerr << "Else\n";
                        else_scope.dump();
                        Scope(latch).dump();
#endif

                        if (joins_old.size() > 1) {
                            //TODO: This is hard-coded!
                            Continuation* next_target = const_cast<Continuation*>(else_new->callee()->as<Continuation>());
                            mem = else_new->param(0);

                            auto load = world_.load(mem, pred_cache);
                            mem = world_.extract(load, (int) 0);

                            auto old_predicate = world_.extract(load, (int) 1);
                            auto additional_predicate = world_.arithop_not(old_predicate);
                            auto new_predicate = world_.binop(ArithOp_or, predicate_false, additional_predicate);
                            auto new_jump = world_.predicated(vec_mask);

                            else_new->jump(new_jump, { mem, new_predicate, next_target, else_new_back }, latch->jump_location());

                            assert(next_target->arg(0)->type()->isa<MemType>());
                            assert(join_is_first_mem);
                            mem = next_target->arg(0);
                            for (size_t j = 0; j < next_target->num_args(); j++) {
                                if (join_is_first_mem && j == 0)
                                    continue;
                                mem = world_.store(mem, join_cache[join_is_first_mem ? j - 1 : j], next_target->arg(j));
                            }
                            next_target->jump(else_new_back, {mem});
                        } else {
                            if (else_new->callee()->isa<Continuation>() && else_new->callee()->as<Continuation>()->is_intrinsic() && else_new->callee()->as<Continuation>()->intrinsic() == Intrinsic::Branch) {
                                //TODO: This is hard-coded!
                                auto old_predicate_true = else_new->arg(0);
                                auto old_predicate_false = world_.arithop_not(old_predicate_true);
                                auto new_predicate_true = world_.binop(ArithOp_and, predicate_false, old_predicate_true);
                                auto new_predicate_false = world_.binop(ArithOp_and, predicate_false, old_predicate_false);

                                auto next_target_true = const_cast<Continuation*>(else_new->arg(1)->as<Continuation>());
                                auto next_target_false = const_cast<Continuation*>(else_new->arg(2)->as<Continuation>());

                                Continuation_MemParam(next_target_true, vector_schedule, block2mem);
                                Continuation_MemParam(next_target_false, vector_schedule, block2mem);

                                auto new_jump_else = world_.predicated(vec_mask);
                                auto new_jump_true = world_.predicated(vec_mask);

                                mem = else_new->param(0);
                                else_new->jump(new_jump_else, {mem, new_predicate_true, next_target_true, next_target_true});

                                mem = next_target_true->arg(0);
                                for (size_t j = 0; j < next_target_true->num_args(); j++) {
                                    if (join_is_first_mem && j == 0)
                                        continue;
                                    mem = world_.store(mem, join_cache[join_is_first_mem ? j - 1 : j], next_target_true->arg(j));
                                }
                                next_target_true->jump(new_jump_true, {mem, new_predicate_false, next_target_false, next_target_false});

                                mem = next_target_false->arg(0);
                                for (size_t j = 0; j < next_target_false->num_args(); j++) {
                                    if (join_is_first_mem && j == 0)
                                        continue;
                                    mem = world_.store(mem, join_cache[join_is_first_mem ? j - 1 : j], next_target_false->arg(j));
                                }
                                next_target_false->jump(else_new_back, {mem});
                            } else {
                                for (auto join_old : joins_old) {
                                    auto join = const_cast<Continuation*>(def2def_[join_old]->as<Continuation>());
                                    assert(join);
                                    for (auto pred : join->preds()) {
                                        if (!else_scope.contains(pred)) {
                                            continue;
                                        }

                                        assert(pred->arg(0)->type()->isa<MemType>());
                                        mem = pred->arg(0);

                                        for (size_t j = 0; j < pred->num_args(); j++) {
                                            if (join_is_first_mem && j == 0)
                                                continue;
                                            mem = world_.store(mem, join_cache[join_is_first_mem ? j - 1 : j], pred->arg(j));
                                        }

                                        pred->jump(else_new_back, {mem});
                                    }
                                }
                            }
                        }

#ifdef DUMP_VECTORIZER
                        Scope(latch).dump();
#endif

                        auto true_elem = world_.one(world_.type_bool());
                        Array<const Def *> elements(vector_width);
                        for (size_t i = 0; i < vector_width; i++) {
                            elements[i] = true_elem;
                        }
                        auto one_predicate = world_.vector(elements);

                        auto new_jump_else = world_.predicated(vec_mask);
                        else_new_back->jump(new_jump_else, { else_new_back->mem_param(), one_predicate, pre_join, pre_join }, latch->jump_location());

                        mem = pre_join->param(0);
                        Array<const Def*> join_params(join->num_params());
                        //join == else is a real option, it will come for you, it will haunt you till the end of days.
                        if (join->num_params() > cache_size) {//We added a mem parameter in the code above!
                            assert(is_mem(join->param(0)));
                            join_is_first_mem = true;
                        }
                        for (size_t i = 0; i < join->num_params(); i++) {
                            if (join_is_first_mem && i == 0)
                                continue;
                            auto load = world_.load(mem, join_cache[join_is_first_mem ? i - 1 : i]);
                            auto value = world_.extract(load, (int) 1);
                            mem = world_.extract(load, (int) 0);
                            join_params[i] = value;
                        }
                        if (join_is_first_mem)
                            join_params[0] = mem;
#ifdef DUMP_VECTORIZER
                        else {
                            std::cerr << "Might need some rewiring of the mem parameter?\n";
                        }
#endif
                        pre_join->jump(join, join_params, latch->jump_location());

#ifdef DUMP_VECTORIZER
                        std::cerr << "Transformed branch\n";
                        Scope(latch).dump();
                        Scope(latch_old).dump();
#endif
                    }

#ifdef DUMP_VECTORIZER
                    Scope(vectorized).dump();
#endif
                }

                {
                    Scope vector_scope(vectorized);
                    Schedule vector_schedule = schedule(vector_scope);
                    for (auto& block : vector_schedule) {
                        const Continuation* cont = block.continuation();
                        const Def* callee = cont->callee();
                        if(callee->name() == "br_vec") {
                            Scope(kerndef).dump();
                            vector_scope.dump();
                            assert(false);
                        }
                    }
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
#ifdef DUMP_VECTORIZER
            std::cerr << "Continuation end\n\n";
#endif
        }

        for (auto succ : cont->succs())
            enqueue(succ);
    }

#ifdef DUMP_VECTORIZER
    std::cout << "Pre Cleanup\n";
    world_.dump();
    std::cout << "Cleanup\n";
#endif
    {
        llvm::TimeRegion clean_time(time_clean);
        world_.cleanup();
    }
#ifdef DUMP_VECTORIZER
    std::cout << "Post Cleanup\n";
    world_.dump();
#endif

    return false;
}

bool vectorize(World& world) {
    VLOG("start vectorizer");
    auto res = Vectorizer(world).run();
    VLOG("end vectorizer");
    return res;
}

}
