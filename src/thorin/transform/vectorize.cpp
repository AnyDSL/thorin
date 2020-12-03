#include "thorin/transform/mangle.h"
#include "thorin/world.h"
#include "thorin/util/log.h"

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

    private:
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

    public:
        DivergenceAnalysis(Continuation* base) : base(base) {};
        void run();
        State getUniform(Def * def);
    };

    World& world_;
    size_t boundary_;
    ContinuationSet done_;
    ContinuationMap<bool> top_level_;
    Def2Def def2def_;
    DivergenceAnalysis * div_analysis_;

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
Vectorizer::DivergenceAnalysis::getUniform(Def * def) {
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

    //TODO: Heavy caching is of the essence.
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
                            for (auto def : scope.defs()) { //TODO: only parameters are verying, not the entire continuation!
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
    if (old_type->isa<PtrType>()) { //TODO: these types might have a length of their own, in this case, I need to extend.
        return world_.ptr_type(old_type->as<PtrType>()->pointee(), 8);
    } else if (old_type->isa<PrimType>()) {
        return world_.type(old_type->as<PrimType>()->primtype_tag(), 8);
    } else {
        return world_.vec_type(old_type, 8);
    }
}

Continuation* Vectorizer::widen_head(Continuation* old_continuation) {
    assert(!def2def_.contains(old_continuation));
    assert(!old_continuation->empty());
    Continuation* new_continuation = old_continuation->stub();
    def2def_[old_continuation] = new_continuation;

    for (size_t i = 0, e = old_continuation->num_params(); i != e; ++i)
        def2def_[old_continuation->param(i)] = new_continuation->param(i);

    return new_continuation;
}

const Def* Vectorizer::widen(const Def* old_def) {
    if (auto new_def = find(def2def_, old_def))
        return new_def;
    else if (!widen_within(old_def))
        return old_def;
    else if (auto old_continuation = old_def->isa_continuation()) {
        auto new_continuation = widen_head(old_continuation);
        widen_body(old_continuation, new_continuation);
        return new_continuation;
    } else if (auto param = old_def->isa<Param>()) {
        widen(param->continuation());
        assert(def2def_.contains(param));
        return def2def_[param];
    } else if (auto param = old_def->isa<Extract>()) {
        auto old_primop = old_def->as<PrimOp>();
        Array<const Def*> nops(old_primop->num_ops());

        //TODO: this is hard coded for extracts after loads.
        //At some point I should distinguish between should- and should-not-vectorize, based on
        //  (a) the types needed to be syntacticly correct
        //  (b) the divergence analysis.

        nops[0] = widen(old_primop->op(0));
        nops[1] = old_primop->op(1);

        auto type = widen(old_primop->type());
        const Def* new_primop;
        if (old_primop->isa<PrimLit>()) {
            Array<const Def*> elements(8);
            for (int i = 0; i < 8; i++) {
                elements[i] = old_primop;
            }
            new_primop = world_.vector(elements, old_primop->debug_history());
        } else {
            new_primop = old_primop->rebuild(nops, type);
        }
        return def2def_[old_primop] = new_primop;
    } else if (auto param = old_def->isa<ArithOp>()) {
        auto old_primop = old_def->as<PrimOp>();
        Array<const Def*> nops(old_primop->num_ops());
        bool any_vector = false;
        for (size_t i = 0, e = old_primop->num_ops(); i != e; ++i) {
            nops[i] = widen(old_primop->op(i));
            if (auto vector = nops[i]->type()->isa<VectorType>())
                any_vector |= vector->is_vector();
        }

        if (any_vector) {
            for (size_t i = 0, e = old_primop->num_ops(); i != e; ++i) {
                if (auto vector = nops[i]->type()->isa<VectorType>())
                    if (vector->is_vector())
                        continue;

                //non-vector element in a vector setting needs to be extended to a vector.
                Array<const Def*> elements(8);
                for (int j = 0; j < 8; j++) {
                    elements[j] = nops[i];
                }
                nops[i] = world_.vector(elements, nops[i]->debug_history());
            }
        }

        auto type = widen(old_primop->type());
        const Def* new_primop;
        if (old_primop->isa<PrimLit>()) {
            Array<const Def*> elements(8);
            for (int i = 0; i < 8; i++) {
                elements[i] = old_primop;
            }
            new_primop = world_.vector(elements, old_primop->debug_history());
        } else {
            new_primop = old_primop->rebuild(nops, type);
        }
        return def2def_[old_primop] = new_primop;
    } else {
        auto old_primop = old_def->as<PrimOp>();
        Array<const Def*> nops(old_primop->num_ops());
        for (size_t i = 0, e = old_primop->num_ops(); i != e; ++i)
            nops[i] = widen(old_primop->op(i));

        auto type = widen(old_primop->type());
        const Def* new_primop;
        if (old_primop->isa<PrimLit>()) {
            Array<const Def*> elements(8);
            for (int i = 0; i < 8; i++) {
                elements[i] = old_primop;
            }
            new_primop = world_.vector(elements, old_primop->debug_history());
        } else {
            new_primop = old_primop->rebuild(nops, type);
        }
        return def2def_[old_primop] = new_primop;
    }
}

void Vectorizer::widen_body(Continuation* old_continuation, Continuation* new_continuation) {
    assert(!old_continuation->empty());

    // fold branch and match
    // TODO find a way to factor this out in continuation.cpp
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

    Defs nargs(nops.skip_front()); // new args of new_continuation
    auto ntarget = nops.front();   // new target of new_continuation

    new_continuation->jump(ntarget, nargs, old_continuation->jump_debug());
}

Continuation *Vectorizer::widen() {
    //Step 1: Dump info.
    //kernel->dump();
    //std::cerr << "\n";

    //Step 2: Create a faithful copy.
    //Continuation *vectorize_start = world_.continuation(kernel->type(), kernel->attributes(), kernel->debug_history());

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
        def2def_[def] = ncontinuation->append_param(def->type()); // TODO reduce

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

    std::queue<const Def*> queue;
    GIDSet<const Def*> done;

#if 0
    std::cerr << "Widening\n";
    Rewriter vectorizer;
    for (size_t i = 0; i < ncontinuation->num_ops(); i++) {
        queue.push(ncontinuation->op(i));
        done.emplace(ncontinuation->op(i));
    }
    while (!queue.empty()) {
        const Def* element = pop(queue);
        element->dump();

        //if div_analysis_->get_result(element_vectorized) == divergent:
        const Def* element_vectorized = widen(element);
        const Def* element_vectorized = nullptr;
        if (element_vectorized) {
          vectorizer.old2new[element] = element_vectorized;
          element_vectorized->dump();
        } else {
          std::cerr << "Not vectorized\n";
        }

        for (auto op : element->ops())
            if (done.emplace(op).second)
                queue.push(op);
    }
    std::cerr << "Widening End\n\n";

    //TODO: This should only find temporary use. We should be able to directly call the vectorized function.
    for (auto use : kernel->copy_uses()) {
        if (auto ucontinuation = use->isa_continuation())
            ucontinuation->update_op(use.index(), ncontinuation);
        else {
            auto primop = use->as<PrimOp>();
            Array<const Def*> nops(primop->num_ops());
            std::copy(primop->ops().begin(), primop->ops().end(), nops.begin());
            nops[use.index()] = ncontinuation;
            auto newprimop = primop->rebuild(nops);
            primop->replace(newprimop);
        }
    }
#endif

    //Scope(ncontinuation).dump();
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
    //TODO: Uniform branches might still need masking. => Predicate generation relevant!
                widen_setup(kerndef);
                auto *vectorized = widen();
                //auto *vectorized = clone(Scope(kerndef));

                delete div_analysis_;

                if (vectorized) {
                    for (auto caller : cont->preds()) {
                        Array<const Def*> args(vectorized->num_params());

                        args[0] = caller->arg(0); //mem
                        //args[1] = caller->arg(1); //width
                        Array<const Def*> defs(8);
                        for (int i = 0; i < 8; i++) {
                            defs[i] = world_.literal_qs32(i, caller->arg(1)->debug_history());
                        }
                        args[1] = world_.vector(defs, caller->arg(1)->debug_history());

                        for (size_t p = 2; p < vectorized->num_params(); p++) {
                            args[p] = caller->arg(p + 1);
                        }

                        caller->jump(vectorized, args, caller->jump_debug());
                    }
                    //const FnType* type = dynamic_cast<const FnType*>(cont->type());
                    //Array<const Type*> args(type->num_ops() - 2);
                    //args[0] = type->op(0);
                    //for (size_t k = 3; k < type->num_ops(); k++)
                    //    args[k - 2] = type->op(k);
                    //const FnType *newtype = world_.fn_type(args);

                    //Continuation::Attributes attributes = cont->attributes();
                    //attributes.intrinsic = Intrinsic::None;

                    //auto *header = world_.continuation(type, attributes, cont->debug_history());
                    //Array<const Def*> params(vectorized->num_params());
                    //params[0] = header->param(0); //mem

                    //params[1] = header->param(1); //width
                    //params[1] = world_.literal_qs32(42, header->debug_history()); //width

                    //params[2] = header->param(1); //return
                    //for (size_t p = 2; p < header->num_params(); p++) {
                    //    params[p + 1] = header->param(p);
                    //}
                    //header->jump(vectorized, params, cont->jump_debug());
                    //
                    //cont->replace(header); //TODO: replace Calls to cont!
                }
            }
            std::cerr << "Continuation end\n\n";
        }
        //if cont is vectorize:
        //find all calls, repeat the rest of this for all of them with the respective set of continuations being used.


        for (auto succ : cont->succs())
            enqueue(succ);
    }


    //world.cleanup();
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
