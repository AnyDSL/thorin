#include "thorin/transform/mangle.h"
#include "thorin/transform/flatten_vectors.h"
#include "thorin/world.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/verify.h"

#include <llvm/Support/Timer.h>
#include <llvm/Support/raw_ostream.h>

#include <iostream>
#include <map>

//#define DUMP_DIV_ANALYSIS
//#define DUMP_WIDEN
//#define DUMP_VECTORIZER

//#define DBG_TIME(v, t) llvm::TimeRegion v(t)
#define DBG_TIME(v, t) (void)(t)

#define DUMP_BLOCK(block) { \
                    Stream s(std::cout); \
                    RecStreamer rec(s, std::numeric_limits<size_t>::max()); \
                    for (auto& block : schedule(Scope(block))) { \
                        rec.conts.push(block); \
                        rec.run(); \
                    } \
                    s.endl(); \
}

namespace thorin {

    //class actually defined in rec_stream.cpp!
class RecStreamer {
public:
    RecStreamer(Stream& s, size_t max)
        : s(s)
        , max(max)
    {}

    void run();
    void run(const Def*);

    Stream& s;
    size_t max;
    unique_queue<ContinuationSet> conts;
    DefSet defs;
};

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
        , kernel(nullptr)
        , current_scope(nullptr)
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
    Scope *current_scope;
    bool widen_within(const Def *);

    void widen_body(Continuation *, Continuation *);
    Continuation* widen_head(Continuation* old_continuation);

    void Continuation_MemParam(Continuation* continuation, Scope& vector_scope, ContinuationMap<const Def*>& block2mem, ContinuationMap<GIDSet<const Def*>>& block2firstmem);
    void analyze_mem(Scope &vector_scope,
                     ContinuationMap<const Def*> &block2mem,
                     ContinuationMap<GIDSet<const Def*>> &block2firstmem);
};

Vectorizer::DivergenceAnalysis::State
Vectorizer::DivergenceAnalysis::getUniform(const Def * def) {
#if 1
    (void)(def);
    return Varying;
#else
    std::cerr << "Get uniform\n";
    std::cerr << def << "\n";
    if (def->isa<Tuple>() && def->op(1)->isa<Continuation>()) {
        std::cerr << "Varying\n";
        return Varying;
    }
    std::cerr << ((uniform[def] == Varying) ? "Varying" : "Uniform") << "\n";
    return uniform[def];
#endif
}

ContinuationSet Vectorizer::DivergenceAnalysis::successors(Continuation * cont) {
    if (loopExits.contains(cont)) {
        return loopExits[cont];
    } else {
        ContinuationSet continues;
        for (auto node : cont->succs())
            if (!(node->is_intrinsic() || node->is_imported()))
                continues.emplace(node);
        return continues;
    }
}

ContinuationSet Vectorizer::DivergenceAnalysis::predecessors(Continuation * cont) {
    ContinuationSet nodes;
    for (auto pre : cont->preds())
        if (!(pre->is_intrinsic() || pre->is_imported()))
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
    const LoopTree<true> &loops = Scope(base).f_cfg().looptree();


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
                if (!(succ->is_intrinsic() || succ->is_imported()))
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
                if (!(succ->is_intrinsic() || succ->is_imported()))
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
            if (succ->is_intrinsic() || succ->is_imported())
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
            int num_relevant = 0;
            for (auto succ : cont->succs()) {
                if (succ->is_imported())
                    continue;
                num_relevant++;
            }
            if (num_relevant > 1 && cont->callee()->isa<Continuation>() && cont->callee()->as<Continuation>()->is_intrinsic()) {
                splitNodes.emplace(cont);
#ifdef DUMP_DIV_ANALYSIS
                cont->dump();
            } else if (num_relevant > 1) {
                std::cerr << "Multiple successors in non-intrinsic node\n";
                cont->dump();
                for (auto succ : cont->succs()) {
                    succ->dump();
                }
                std::cerr << "Of which " << num_relevant << " are cosidered\n";
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
            if (succ->is_intrinsic() ||  succ->is_imported())
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
            for (auto key : keys)
                key->dump();
            std::cerr << "Predecessors End\n";
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
                for (auto pre : plabs)
                    pre->dump();
                std::cerr << "Previous End\n";
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
                for (auto succ : successors(cont))
                    succ->dump();
                std::cerr << "Successors End\n";
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
        for (auto label : LabelMap) {
            label.first->dump();
            std::cerr << "-> ";
            label.second->dump();
        }
        std::cerr << "Joins:\n";
        for (auto join : Joins)
            join->dump();
        std::cerr << "End\n";
#endif
    }

    //Step 2.3: Sort Split-Joins in a hirarchy
    for (auto it : relJoins) {
        for (auto it_inner : relJoins) {
            if (it_inner.first == it.first)
                continue;
            if (!dominatedBy[it_inner.first].contains(it.first))
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
    Scope base_scope(base);
    for (auto def : base_scope.defs())
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
            for (auto param : join->params()) {
#ifdef DUMP_DIV_ANALYSIS
                std::cerr << "Mark param varying\n";
                param->dump();
#endif
                uniform[param] = Varying;
                def_queue.push(param);
            }
        }
    }

#ifdef DUMP_DIV_ANALYSIS
    std::cerr << "RelJoins\n";
    for (auto it : relJoins) {
        it.first->dump();
        for (auto s : it.second)
            s->dump();
    }
    std::cerr << "SplitParrents\n";
    for (auto it : splitParrents) {
        it.first->dump();
        for (auto s : it.second)
            s->dump();
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
                    auto target = cont->op(0)->isa_continuation();
                    if (is_op && target && target->is_intrinsic() && opnum == 1 && relJoins.find(cont) != relJoins.end()) {
                        ContinuationSet joins = relJoins[cont];
                        for (auto join : joins) {
                            for (auto param : join->params()) {
#ifdef DUMP_DIV_ANALYSIS
                                std::cerr << "Mark param varying\n";
                                param->dump();
#endif
                                uniform[param] = Varying;
                                def_queue.push(param);
                            }
                        }
                    } else if (target && is_op) {
                        if (target->is_imported()) {
                            auto actualtarget = const_cast<Continuation*>(cont->op(cont->num_ops() - 1)->isa<Continuation>());
                            assert(actualtarget);
                            for (auto param : actualtarget->params()) {
#ifdef DUMP_DIV_ANALYSIS
                                std::cerr << "Mark param varying\n";
                                param->dump();
#endif
                                uniform[param] = Varying;
                                def_queue.push(param);
                            }
                        }
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
                            for (auto s : it.second)
                                s->dump();
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
    if (auto primtype = old_type->isa<PrimType>()) {
        assert(primtype->length() == 1);
        return world_.prim_type(primtype->primtype_tag(), vector_width);
    } else if (auto ptrtype = old_type->isa<PtrType>()) {
        assert(ptrtype->length() == 1);
        return world_.ptr_type(ptrtype->pointee(), vector_width);
    } else {
        return world_.vec_type(old_type, vector_width);
    }
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

    assert(new_continuation);
    def2def_[old_continuation] = new_continuation;

    for (size_t i = 0, e = old_continuation->num_params(); i != e; ++i) {
        assert(new_continuation->param(i));
        def2def_[old_continuation->param(i)] = new_continuation->param(i);
    }

    return new_continuation;
}

const Def* Vectorizer::widen(const Def* old_def) {
#ifdef DUMP_WIDEN
    std::cout << "Widen\n";
    old_def->dump();
#endif

    if (def2def_.contains(old_def)) {
#ifdef DUMP_WIDEN
        std::cout << "Found\n";
#endif
        auto new_def = def2def_[old_def];
        return new_def;
    } else if (!widen_within(old_def)) {
#ifdef DUMP_WIDEN
        std::cout << "NWithin\n";
#endif
        if (auto cont = old_def->isa_continuation()) {
            if (cont->is_intrinsic() && cont->intrinsic() == Intrinsic::Match) {
                auto type = old_def->type();
                auto match = world_.match(widen(type->op(0)), type->num_ops() - 2);
                assert(match);
                return def2def_[old_def] = match;
            }
            if (cont->is_intrinsic() && cont->intrinsic() == Intrinsic::Branch) {
                auto branch = world_.continuation(world_.fn_type({widen(world_.type_bool()), world_.fn_type(), world_.fn_type()}), Intrinsic::Branch, {"br_vec"});
                assert(branch);
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
        //Make a copy!
        Array<const Def*> nops(old_def->num_ops());

        for (unsigned i = 0; i < old_def->num_ops(); i++) {
            nops[i] = (widen(old_def->op(i))); //These should all be uniform as well.
        }

        auto r = old_def->as<PrimOp>()->rebuild(nops);
        return r; //TODO: this def could contain a continuation inside a tuple for match cases!
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
        if (auto vectype = nops[0]->type()->isa<VectorType>(); vectype && vectype->is_vector())
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
        assert(new_primop);
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
        assert(new_primop);
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
            if (nops[i]->type()->isa<VariantVectorType>())
                any_vector = true;
        }

        if (any_vector) {
            for (size_t i = 0, e = param->num_ops(); i != e; ++i) {
                if (auto vector = nops[i]->type()->isa<VectorType>())
                    if (vector->is_vector())
                        continue;
                if (nops[i]->type()->isa<VariantVectorType>())
                    continue;
                if (nops[i]->type()->isa<MemType>())
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
        assert(new_primop);
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
            if (auto vectype = nops[i]->type()->isa<VectorType>(); vectype && vectype->is_vector())
                any_vector = true;
            if (nops[i]->type()->isa<VariantVectorType>())
                any_vector = true;
        }

        if (any_vector && (old_primop->isa<BinOp>() || old_primop->isa<Select>() || old_primop->isa<StructAgg>() || old_primop->isa<Access>())) {
            for (size_t i = 0, e = old_primop->num_ops(); i != e; ++i) {
                if (auto vectype = nops[i]->type()->isa<VectorType>(); vectype && vectype->is_vector())
                    continue;
                if (nops[i]->type()->isa<VariantVectorType>())
                    continue;
                if (nops[i]->type()->isa<MemType>())
                    continue;
                Array<const Def*> elements(vector_width);
                for (size_t j = 0; j < vector_width; j++)
                    elements[j] = nops[i];
                nops[i] = world_.vector(elements, nops[i]->debug_history());
            }
        }

        if (old_def->isa<Slot>()) {
            //force creation of a vectorized slot
            any_vector = true;
        }

        const Type* type;
        if (any_vector)
          type = widen(old_primop->type());
        else
          type = old_primop->type();

        const Def* new_primop;

        if (old_primop->isa<PrimLit>()) {
            assert(false && "Primlits are uniform");
        } else {
            new_primop = old_primop->rebuild(nops, type);
        }
        if (old_def->isa<Slot>()) {
            assert(new_primop->type() == type);
            auto vectype = new_primop->type()->isa<VectorType>(); 
            assert(vectype && vectype->is_vector());
        }
        assert(new_primop);
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
                    return new_continuation->jump(widen(cont), {}, old_continuation->debug());
                }
                break;
            }
            case Intrinsic::Match:
                if (old_continuation->num_args() == 2)
                    return new_continuation->jump(widen(old_continuation->arg(1)), {}, old_continuation->debug());

                if (auto lit = widen(old_continuation->arg(0))->isa<PrimLit>()) {
                    for (size_t i = 2; i < old_continuation->num_args(); i++) {
                        auto new_arg = widen(old_continuation->arg(i));
                        if (world_.extract(new_arg, 0_s)->as<PrimLit>() == lit)
                            return new_continuation->jump(world_.extract(new_arg, 1), {}, old_continuation->debug());
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
            if (auto vectype = nops[1]->type()->isa<VectorType>(); !vectype || !vectype->is_vector()) {
                ntarget = old_continuation->op(0);
            }
        }
        if (callee->is_imported()) {
            auto old_fn_type = callee->type()->as<FnType>();
            Array<const Type*> ops(old_fn_type->num_ops());
            bool anyvector = false;
            for (size_t i = 0; i < old_fn_type->num_ops(); i++) {
                ops[i] = nops[i + 1]->type(); //TODO: this feels like a bad hack. At least it's working for now.
                if (auto vectype = ops[i]->isa<VectorType>(); vectype && vectype->is_vector())
                    anyvector = true;
            }
            Debug de = callee->debug();

            if (de.name.rfind("rv_", 0) == 0) {
                std::cerr << "RV intrinsic: " << de.name << "\n";
                assert(false);
            }

            if (anyvector) {
                if (de.name == "llvm.exp.f32")
                    de.name = "llvm.exp.v8f32"; //TODO: Use vectorlength to find the correct intrinsic.
                else if (de.name == "llvm.exp.f64")
                    de.name = "llvm.exp.v8f64";
                else if (de.name == "llvm.sqrt.f32")
                    de.name = "llvm.sqrt.v8f32";
                else if (de.name == "llvm.sqrt.f64")
                    de.name = "llvm.sqrt.v8f64";
                else if (de.name == "llvm.sin.f32")
                    de.name = "llvm.sin.v8f32";
                else if (de.name == "llvm.sin.f64")
                    de.name = "llvm.sin.v8f64";
                else if (de.name == "llvm.cos.f32")
                    de.name = "llvm.cos.v8f32";
                else if (de.name == "llvm.cos.f64")
                    de.name = "llvm.cos.v8f64";
                else if (de.name == "llvm.minnum.f32")
                    de.name = "llvm.minnum.v8f32";
                else if (de.name == "llvm.minnum.f64")
                    de.name = "llvm.minnum.v8f64";
                else if (de.name == "llvm.floor.f32")
                    de.name = "llvm.floor.v8f32";
                else if (de.name == "llvm.floor.f64")
                    de.name = "llvm.floor.v8f64";
                else {
                    std::cerr << "Not supported: " << de.name << "\n";
                    assert(false && "Import not supported in vectorize.");
                }

                ntarget = world_.continuation(world_.fn_type(ops), callee->attributes(), de);
            } else {
                ntarget = world_.continuation(world_.fn_type(ops), callee->attributes(), de);
            }
            assert(ntarget);
            def2def_[callee] = ntarget;
        }
    }

    Scope scope(kernel);

    if (old_continuation->op(0)->isa<Continuation>() && scope.contains(old_continuation->op(0))) {
        auto oldtarget = old_continuation->op(0)->as<Continuation>();

        for (size_t i = 0; i < nops.size() - 1; i++) {
            auto arg = nops[i + 1];
            if (!is_mem(arg) &&
                    (!arg->type()->isa<VectorType>() ||
                     !arg->type()->as<VectorType>()->is_vector()) && //TODO: This is not correct.
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
            if ((!nargs[i]->type()->isa<VectorType>() || !nargs[i]->type()->as<VectorType>()->is_vector()) &&
                    ntarget->isa<Continuation>() &&
                    ntarget->as<Continuation>()->param(i)->type()->isa<VectorType>() &&
                    ntarget->as<Continuation>()->param(i)->type()->as<VectorType>()->is_vector()) { //TODO: base this on divergence analysis
                Array<const Def*> elements(vector_width);
                for (size_t j = 0; j < vector_width; j++)
                    elements[j] = nargs[i];
                nargs[i] = world_.vector(elements, nargs[i]->debug_history());
            }
    }

    new_continuation->jump(ntarget, nargs, old_continuation->debug());
#ifdef DUMP_WIDEN
    std::cout << "Jump\n";
#endif
}

Continuation *Vectorizer::widen() {
    Continuation *ncontinuation;

    // create new_entry - but first collect and specialize all param types
    std::vector<const Type*> param_types;
    for (size_t i = 0, e = kernel->num_params(); i != e; ++i) {
        if (i == 1) {
            param_types.emplace_back(widen(kernel->param(i)->type()));
        } else {
            param_types.emplace_back(kernel->param(i)->type());
        }
    }

    auto fn_type = world_.fn_type(param_types);
    ncontinuation = world_.continuation(fn_type, kernel->debug_history());

    // map value params
    assert(kernel);
    def2def_[kernel] = kernel;
    for (size_t i = 0, j = 0, e = kernel->num_params(); i != e; ++i) {
        auto old_param = kernel->param(i);
        auto new_param = ncontinuation->param(j++);
        assert(new_param);
        def2def_[old_param] = new_param;
        new_param->debug().name = old_param->name();
    }

    // mangle filter
    if (!kernel->filter().empty()) {
        Array<const Def*> new_filter(ncontinuation->num_params());
        size_t j = 0;
        for (size_t i = 0, e = kernel->num_params(); i != e; ++i) {
            new_filter[j++] = widen(kernel->filter(i));
        }

        for (size_t e = ncontinuation->num_params(); j != e; ++j)
            new_filter[j] = world_.literal_bool(false, Debug{});

        ncontinuation->set_filter(new_filter);
    }

    widen_body(kernel, ncontinuation);

    return ncontinuation;
}

void Vectorizer::widen_setup(Continuation* kern) {
    kernel = kern;
    if (current_scope)
        delete current_scope;
    current_scope = new Scope(kernel);
}

bool Vectorizer::widen_within(const Def* def) {
    return current_scope->contains(def);
}

void Vectorizer::Continuation_MemParam(Continuation* continuation, Scope &vector_scope, ContinuationMap<const Def*>& block2mem, ContinuationMap<GIDSet<const Def*>>& block2firstmem) {
    //std::cerr << "\n";
    //std::cerr << "Adding mem to " << continuation->to_string() << "\n";

    auto& cfg = vector_scope.f_cfg();
    auto& domtree = cfg.domtree();
    Continuation* old_continuation = nullptr;
    for (auto it : def2def_) { //TODO: Make this FAST!
        if (it.second == continuation)
            old_continuation = const_cast<Continuation*>(it.first->as<Continuation>());
    }
    assert(old_continuation);

    for (auto param : continuation->params()) {
        assert(!is_mem(param));
    }
    auto continuation_mem = continuation->append_param(world_.mem_type()); //TODO: prepend instead of append?

    auto idom = domtree.idom(cfg[continuation]);

    auto mem_last = block2mem[idom->continuation()];
    assert(mem_last);

    auto continuation_first_mem_set = block2firstmem[continuation];

    //if(!continuation_first_mem_set.size()) {
        //std::cerr << "No Mem uses\n";
    //}

    ContinuationSet toUpdate;
    for (auto it : block2mem) {
        if (it.second == mem_last) {
            if (it.first == continuation || domtree.least_common_ancestor(cfg[it.first], cfg[continuation])->continuation() == continuation) {
                //std::cerr << "Relevant block2mem: " << it.first->to_string() << "\n";
                toUpdate.emplace(it.first);
            }
        }
    }
    for (auto it : toUpdate) {
        block2mem[it] = continuation_mem;
    }

    for (auto continuation_first_mem : continuation_first_mem_set) {
        //std::cerr << "first mem used by " << continuation_first_mem->to_string() << "\n";
        //std::cerr << "This uses memory " << mem_last->to_string() << " from " << idom->continuation()->to_string() << "\n";

        if (auto using_cont = continuation_first_mem->isa_continuation()) {
            unsigned opnum = 0;
            for (auto out : using_cont->ops()) {
                if  (is_mem(out)) {
                    break;
                }
                opnum++;
            }

            using_cont->unset_op(opnum);
            using_cont->set_op(opnum, continuation_mem);
        } else {
            //Replace mem parameter in this particular instance.
            assert(continuation_first_mem->isa<MemOp>());
            assert(is_mem(continuation_first_mem->as<MemOp>()->op(0)));
            auto memop = const_cast<MemOp*>(continuation_first_mem->as<MemOp>());
            memop->unset_op(0);
            memop->set_op(0, continuation_mem);
        }

        ContinuationSet toUpdate;

        //TODO: Update block2firstmem accordingly.
        for (auto it : block2firstmem) {
            if (it.second.contains(continuation_first_mem)) {
                if (it.first != continuation && domtree.least_common_ancestor(cfg[it.first], cfg[continuation])->continuation() != continuation) {
                    //std::cerr << "Relevant block2firstmem: " << it.first->to_string() << "\n";
                    toUpdate.emplace(it.first);
                }
            }
        }

        for (auto it : toUpdate) {
            GIDSet<const Def*> firstmem = block2firstmem[it];
            firstmem.erase(continuation_first_mem);
            block2firstmem[it] = firstmem;
        }
    }
}

void Vectorizer::analyze_mem(Scope &vector_scope,
                        ContinuationMap<const Def*> &block2mem,
                        ContinuationMap<GIDSet<const Def*>> &block2firstmem) {
    for (auto n : vector_scope.f_cfg().post_order()) {
        if (n->continuation() == world_.end_scope())
            continue;
        const Def* out_mem = nullptr;
        for (auto out : n->continuation()->ops())
            if  (is_mem(out)) {
                out_mem = out;
                break;
            }
        if (out_mem) {
            block2mem[n->continuation()] = out_mem;
        } else {
            std::queue <const Def*> queue;
            Def2Def coloring;
            DefSet current_colors;
            for (auto succ : n->continuation()->succs()) {
                const Def* out_mem = block2mem[succ];
                if(!out_mem) //Can happen if succ is not in cfg.
                    continue;
                assert(out_mem);
                queue.push(out_mem);
                coloring[out_mem] = out_mem;
                current_colors.emplace(out_mem);
            }
            while (!queue.empty() && current_colors.size() > 1) {
                const Def* next = pop(queue);
                assert(is_mem(next));

                if (next->isa<Param>())
                    continue;

                const Def* pred;
                if (auto extract = next->isa<Extract>())
                    pred = extract->agg()->as<MemOp>()->mem();
                else
                    pred = next->as<MemOp>()->mem();

                //if (pred->isa<Param>())
                //    continue;

                assert(is_mem(pred));

                if (coloring[pred] && ! current_colors.contains(coloring[pred])) { //Treat this node as if it was not colored.
                    coloring[pred] = coloring[next];
                    queue.push(pred);
                } else if (coloring[pred] && coloring[pred] != pred && coloring[pred] != coloring[next]) {
                    auto old_color = coloring[pred];
                    assert(old_color);
                    auto next_color = coloring[next];
                    assert(next_color);

                    current_colors.erase(old_color);
                    current_colors.erase(next_color);
                    current_colors.emplace(pred);

                    coloring[pred] = pred;
                    queue.push(pred);
                } else if (coloring[pred] && coloring[pred] == pred) {
                    //We reached a previous join node. End current branch.
                    auto next_color = coloring[next];
                    assert(next_color);
                    current_colors.erase(next_color);
                } else if (! coloring[pred]){
                    coloring[pred] = coloring[next];
                    queue.push(pred);
                }
            }
            assert(current_colors.size() == 1);
            block2mem[n->continuation()] = *current_colors.begin();
        }
    }

    for (auto n : vector_scope.f_cfg().post_order()) {
        if (n->continuation() == world_.end_scope() || n->continuation()->is_intrinsic())
            continue;
        const Def* mem_param = nullptr;
        for (auto in : n->continuation()->params())
            if  (is_mem(in)) {
                mem_param = in;
                break;
            }

        auto last_mem = mem_param;
        if (!last_mem) {
            auto& cfg = vector_scope.f_cfg();
            auto& domtree = cfg.domtree();
            auto idom = domtree.idom(cfg[n->continuation()]);
            last_mem = block2mem[idom->continuation()];
        }

        const Def* result_mem = block2mem[n->continuation()];

        const Def* out_mem = nullptr;
        unsigned opnum = 0;
        for (auto out : n->continuation()->ops()) {
            if  (is_mem(out)) {
                out_mem = out;
                break;
            }
            opnum++;
        }

        if (result_mem != last_mem) {
            const Def* continuation_first_mem = result_mem;

            while (true) {
                bool contained = false;
                for (auto use : last_mem->copy_uses()) {
                    if (use.def() == continuation_first_mem) {
                        contained = true;
                        break;
                    }
                }
                if (contained)
                    break;

                if (auto extract = continuation_first_mem->isa<Extract>()) {
                    continuation_first_mem = extract->agg();
                } else if (auto memop = continuation_first_mem->isa<MemOp>()) {
                    continuation_first_mem = memop->mem();
                } else {
                    THORIN_UNREACHABLE;
                }
            }

            block2firstmem[n->continuation()] = { continuation_first_mem };
        } else if (out_mem) {
            assert(out_mem == result_mem);
            block2firstmem[n->continuation()] = { n->continuation() };
        } else {
            // collect firstmem for children. Defined by postorder.
            GIDSet<const Def*> childsfirstmem;
            for (auto child : n->continuation()->succs()) {
                if (child == world_.end_scope() || child->is_intrinsic())
                    continue;
                auto childfirstmem = block2firstmem[child];
                //if(!childfirstmem.size()) {
                //    child->dump();
                //}
                assert(childfirstmem.size());
                for (auto firstmem : childfirstmem)
                    childsfirstmem.emplace(firstmem);
            }
            assert(childsfirstmem.size());
            block2firstmem[n->continuation()] = childsfirstmem;
        }
    }
}


bool Vectorizer::run() {
#ifdef DUMP_VECTORIZER
    world_.dump();
#endif

    DBG_TIME(vregion, time);

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
            cont->dump();
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
                    Continuation * ret_join = world_.continuation(ret_type, Debug("shim"));
                    ret_param->replace(ret_join);

                    Array<const Def*> args(ret_join->num_params());
                    for (size_t i = 0, e = ret_join->num_params(); i < e; i++) {
                        args[i] = ret_join->param(i);
                    }

                    ret_join->jump(ret_param, args);
#ifdef DUMP_VECTORIZER
                    DUMP_BLOCK(kerndef);
#endif
                }
    //Warning: Will fail to produce meaningful results or rightout break the program if kerndef does not dominate its subprogram
                {
                    DBG_TIME(div_time, time_div);
                    div_analysis_ = new DivergenceAnalysis(kerndef);
                    div_analysis_->run();
                }

    //Task 2: Widening
                Continuation* vectorized;
                {
                    DBG_TIME(widen_time, time_widen);
                    widen_setup(kerndef);
                    vectorized = widen();
                }
                //auto *vectorized = clone(Scope(kerndef));
                assert(vectorized);
                def2def_[kerndef] = vectorized;

#ifdef DUMP_VECTORIZER
                DUMP_BLOCK(vectorized);
#endif
                {
    //Task 3: Linearize divergent controll flow
                DBG_TIME(lin_time, time_lin);

                std::queue <Continuation*> split_queue;
                GIDMap<Continuation*, ContinuationSet> encloses_splits;
                for (auto it : div_analysis_->relJoins) {
                    if (div_analysis_->splitParrents.contains(it.first) && div_analysis_->splitParrents[it.first].size()) {
                        for (auto encloses : div_analysis_->splitParrents[it.first]) {
                            encloses_splits[encloses].emplace(it.first);
                        }
                    } else {
                        split_queue.push(it.first);
                    }
                }

                //DUMP_BLOCK(vectorized);

                //for (auto it : div_analysis_->relJoins) {
                while (!split_queue.empty()) {
                    Continuation* latch_old = pop(split_queue);
                    ContinuationSet joins_old = div_analysis_->relJoins[latch_old];
                    assert (!div_analysis_->splitParrents.contains(latch_old));
                    Continuation * latch = const_cast<Continuation*>(def2def_[latch_old]->as<Continuation>());
                    assert(latch);

#ifdef DUMP_VECTORIZER
                    DUMP_BLOCK(latch_old);
                    for (auto join : joins_old)
                        join->dump();
                    DUMP_BLOCK(latch);
#endif

                    Continuation *join;
                    if (joins_old.size() == 0) {
                        assert(false && "This should no longer happen");
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
                            if(!div_analysis_->reachableBy[join_old].contains(join_it)) {
                                join_it->dump();
                                join_old->dump();
                                std::cerr << "Going to fail\n";
                            }
                            assert(div_analysis_->reachableBy[join_old].contains(join_it));
                        }
                        join = const_cast<Continuation*>(def2def_[join_old]->as<Continuation>());
                        assert(join);
                    }

#ifdef DUMP_VECTORIZER
                    DUMP_BLOCK(vectorized);
#endif

                    //cases according to latch: match and branch.
                    //match:
                    auto cont = latch->op(0)->isa_continuation();
                    if (cont && cont->is_intrinsic() && cont->intrinsic() == Intrinsic::Match && latch->arg(0)->type()->isa<VectorType>() && latch->arg(0)->type()->as<VectorType>()->is_vector()) {
                        assert(joins_old.size() <= 1 && "no complex controll flow match");

                        const Def* current_frame = nullptr;

                        Scope vector_scope(vectorized); // This might still be borked.
                        Schedule vector_schedule = schedule(vector_scope);

                        //auto& cfg = vector_scope.f_cfg();
                        //auto& domtree = cfg.domtree();
                        //TODO: This code is currently duplicated between branch and match!
                        ContinuationMap<const Def*> block2mem;
                        ContinuationMap<GIDSet<const Def*>> block2firstmem;

                        analyze_mem(vector_scope, block2mem, block2firstmem);

#ifdef DUMP_VECTORIZER
                        DUMP_BLOCK(vectorized);
                        for (auto it : block2mem) {
                            std::cout << "\n";
                            it.first->dump();
                            it.second->dump();
                        }
#endif

                        auto vec_mask = widen(world_.type_bool());
                        auto variant_index = latch->arg(0);

                        //Add memory parameters to make rewiring easier.
                        for (auto split_old : encloses_splits[latch_old]) {
                            for (size_t i = 1; i < split_old->num_args(); i++) {
                                const Def * old_case = split_old->arg(i);
                                if (i != 1) {
                                    assert(old_case->isa<Tuple>());
                                    old_case = old_case->as<Tuple>()->op(1);
                                }
                                Continuation * new_case = const_cast<Continuation*>(def2def_[old_case]->as<Continuation>());
                                assert(new_case);
                                Continuation_MemParam(new_case, vector_scope, block2mem, block2firstmem);
                            }
                        }

                        {//Also extend everything related to the top-level latch.
                            for (size_t i = 1; i < latch_old->num_args(); i++) {
                                const Def * old_case = latch_old->arg(i);
                                if (i != 1) {
                                    assert(old_case->isa<Tuple>());
                                    old_case = old_case->as<Tuple>()->op(1);
                                }
                                Continuation * new_case = const_cast<Continuation*>(def2def_[old_case]->as<Continuation>());
                                assert(new_case);
                                Continuation_MemParam(new_case, vector_scope, block2mem, block2firstmem);
                            }
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
                        {
                            const Def *mem = block2mem[latch];
                            assert(mem);
#ifndef OLDALLOC
                            if (cache_size && !current_frame) {
                                auto mem = vectorized->mem_param();
                                auto enter = world_.enter(mem)->as<Enter>();
                                current_frame = enter->out_frame();
                                auto newmem = enter->out_mem();
                                for (auto use : mem->copy_uses()) {
                                    auto use_inst = use.def();
                                    if (use_inst == enter)
                                        continue;
                                    assert(use_inst);
                                    int index = use.index();
                                    Def* olduse = const_cast<Def*>(use_inst);
                                    olduse->unset_op(index);
                                    olduse->set_op(index, newmem);
                                }
                                if (block2mem[vectorized] == mem)
                                    block2mem[vectorized] = newmem;
                            }
#endif

                            for (size_t i = 0; i < cache_size; i++) {
#ifdef OLDALLOC
                                auto t = world_.alloc(join->param(join_is_first_mem ? i + 1: i)->type(), mem, Debug("join_cache_match"));
                                mem = world_.extract(t, (int) 0);
                                join_cache[i] = world_.extract(t, 1);
#else
                                assert(current_frame);
                                auto t = world_.slot(join->param(join_is_first_mem ? i + 1: i)->type(), current_frame, Debug("join_cache_match"));
                                join_cache[i] = t;
#endif
                            }

                            block2mem[latch] = mem;
                        }
                        //block2mem is considered incorrect at this point, as latch does not use the mem parameter.

                        //Larger cases might be possible?
                        assert(encloses_splits[latch_old].size() <= 1);

                        Array<const Continuation*> splits(encloses_splits[latch_old].size()+1);
                        splits[0] = latch_old;
                        size_t i = 1;
                        for (auto split_old : encloses_splits[latch_old])
                            splits[i++] = split_old;
                        std::sort (splits.begin(), splits.end(), [&](const Continuation *ac, const Continuation *bc) {
                                Continuation *a = const_cast<Continuation*>(ac);
                                Continuation *b = const_cast<Continuation*>(bc);
                                return div_analysis_->dominatedBy[a].contains(b);
                            });

                        for (size_t current_split_index = 0; current_split_index < splits.size(); current_split_index++) {
                            auto split_old = splits[current_split_index];
                            auto split = const_cast<Continuation*>(def2def_[split_old]->as<Continuation>());
                            assert(split);

                            Array<const Def*> local_predicates(split->num_args() - 1);
                            for (size_t i = 1; i < split->num_args() - 1; i++) {
                                auto elem = split->arg(i + 1);
                                auto val = elem->as<Tuple>()->op(0)->as<PrimLit>();
                                Array<const Def *> elements(vector_width);
                                for (size_t i = 0; i < vector_width; i++) {
                                    elements[i] = val;
                                }
                                auto val_vec = world_.vector(elements);
                                auto pred = world_.cmp(Cmp_eq, variant_index, val_vec);
                                local_predicates[i] = pred;
                            }
                            local_predicates[0] = local_predicates[1];
                            for (size_t i = 2; i < split->num_args() - 1; i++) {
                                local_predicates[0] = world_.binop(ArithOp_or, local_predicates[0], local_predicates[i]);
                            }
                            local_predicates[0] = world_.arithop_not(local_predicates[0]);

                            Array<const Def*> split_predicates = local_predicates;

                            const Def * otherwise_old = split_old->arg(1);
                            Continuation * otherwise = const_cast<Continuation*>(def2def_[otherwise_old]->as<Continuation>());
                            assert(otherwise);

                            Continuation * case_old = const_cast<Continuation*>(otherwise_old->as<Continuation>());
                            Continuation * current_case = otherwise;
                            Continuation * case_back = world_.continuation(world_.fn_type({world_.mem_type()}), Debug("otherwise_back"));
                        
                            auto new_jump_split = world_.predicated(vec_mask);
                            assert(current_case != case_back);
                            { //mem scope
                                const Def *mem = block2mem[split];

                                split->jump(new_jump_split, { mem, split_predicates[0], current_case, case_back }, split->debug());
                            }

                            Continuation * pre_join = world_.continuation(world_.fn_type({world_.mem_type()}), Debug("match_merge"));
                            for (size_t i = 2; i < split_old->num_args() + 1; i++) {
                                Continuation *next_case_old = nullptr;
                                Continuation *next_case = nullptr;
                                Continuation *next_case_back = nullptr;

                                if (i < split_old->num_args()) {
                                    next_case_old = const_cast<Continuation*>(split_old->arg(i)->as<Tuple>()->op(1)->as<Continuation>());
                                    next_case = const_cast<Continuation*>(def2def_[next_case_old]->as<Continuation>());
                                    next_case_back = world_.continuation(world_.fn_type({world_.mem_type()}), Debug("case_back"));
                                } else {
                                    next_case = pre_join;
                                    next_case_back = pre_join;
                                }

                                assert(next_case);
                                assert(next_case_back);

                                bool case_back_has_jump = false;

                                Continuation* join_old = *joins_old.begin();

                                for (auto pred_old : join_old->preds()) {
                                    if (!div_analysis_->dominatedBy[pred_old].contains(const_cast<Continuation*>(case_old)) && pred_old != case_old)
                                        continue;
                                    
                                    auto pred = const_cast<Continuation*>(def2def_[pred_old]->as<Continuation>());
                                    assert(pred);

                                    assert(pred->arg(0)->type()->isa<MemType>());
                                    { //mem scope
                                        const Def* mem = block2mem[pred];
                                        assert(mem);

                                        for (size_t j = 1; j < pred->num_args(); j++) {
                                            mem = world_.store(mem, join_cache[join_is_first_mem ? j - 1 : j], pred->arg(j));
                                        }

                                        pred->jump(case_back, { mem });
                                        block2mem[pred] = mem;
                                    }

                                    auto new_jump_case = world_.predicated(vec_mask);
                                    const Def* predicate;
                                    if (i < split_old->num_args())
                                        predicate = split_predicates[i - 1];
                                    else {
                                        auto true_elem = world_.one(world_.type_bool());
                                        Array<const Def *> elements(vector_width);
                                        for (size_t i = 0; i < vector_width; i++) {
                                            elements[i] = true_elem;
                                        }
                                        predicate = world_.vector(elements);
                                    }

                                    if (next_case == next_case_back) {
                                        auto true_elem = world_.one(world_.type_bool());
                                        Array<const Def *> elements(vector_width);
                                        for (size_t i = 0; i < vector_width; i++) {
                                            elements[i] = true_elem;
                                        }
                                        auto one_predicate = world_.vector(elements);
                                        case_back->jump(new_jump_case, { case_back->mem_param(), one_predicate, next_case, next_case_back }, split->debug());
                                        case_back_has_jump = true;
                                    } else {
                                        case_back->jump(new_jump_case, { case_back->mem_param(), predicate, next_case, next_case_back }, split->debug());
                                        case_back_has_jump = true;
                                    }
                                }

                                assert(case_back_has_jump);

                                current_case = next_case;
                                case_old = next_case_old;
                                case_back = next_case_back;

                            }

                            { //mem scope
                                const Def* mem = pre_join->mem_param();

                                Array<const Def*> join_params(join->num_params());
                                for (size_t i = 1; i < join->num_params(); i++) {
                                    auto load = world_.load(mem, join_cache[join_is_first_mem ? i - 1 : i]);
                                    auto value = world_.extract(load, (int) 1);
                                    mem = world_.extract(load, (int) 0);
                                    join_params[i] = value;
                                }
                                join_params[0] = mem;
                                pre_join->jump(join, join_params, split->debug());
                            }
                        }
                    }

#ifdef DUMP_VECTORIZER
                    DUMP_BLOCK(vectorized);
#endif

                    if (cont && cont->is_intrinsic() && cont->intrinsic() == Intrinsic::Branch && latch->arg(0)->type()->isa<VectorType>() && latch->arg(0)->type()->as<VectorType>()->is_vector()) {
                        //TODO: Enclosed if-blocks require special care to use the correct predicates throughout execution.

                        const Def* current_frame = nullptr;

                        //assert (joins_old.size() <= 1 && "no complex controll flow branch");
                        if (joins_old.size() > 1) {
                            //This should only occur when the condition contains an "and" or "or" instruction.
                            //TODO: try to assert this somehow?
                            assert(joins_old.size() == 2 && "Only this case is supported for now.");
                        }

                        {
                            const Continuation * then_old = latch_old->arg(1)->as<Continuation>();
                            Continuation * then_new = const_cast<Continuation*>(def2def_[then_old]->as<Continuation>());
                            assert(then_new);

                            const Continuation * else_old = latch_old->arg(2)->as<Continuation>();
                            Continuation * else_new = const_cast<Continuation*>(def2def_[else_old]->as<Continuation>());
                            assert(else_new);

#ifdef DUMP_VECTORIZER
                            std::cerr << "\n";
#endif

                            if (joins_old.size() > 1) {
                                assert(then_old->succs().size() == 3 && "Additional branching should occur on the 'then' side.");
                                assert(else_old->succs().size() == 1 && "Additional branching should occur on the 'then' side.");

#ifdef DUMP_VECTORIZER
                                for (auto elem : joins_old)
                                    elem->dump();
                                for (auto elem : then_old->as<Continuation>()->succs())
                                    elem->dump();
#endif

                                assert(then_old->as<Continuation>()->succs()[2]->succs().size() == 1);
                                assert(joins_old.contains(then_old->as<Continuation>()->succs()[2]->succs()[0]));
                            }
                        }

#ifdef DUMP_VECTORIZER
                        std::cerr << "Pre transformation\n";
                        DUMP_BLOCK(latch_old);
                        DUMP_BLOCK(latch);
#endif

                        Scope vector_scope(vectorized); //TODO: Might still be borked.
                        Schedule vector_schedule = schedule(vector_scope);

                        //auto& cfg = vector_scope.f_cfg();
                        //auto& domtree = cfg.domtree();
                        ContinuationMap<const Def*> block2mem;
                        ContinuationMap<GIDSet<const Def*>> block2firstmem;

                        analyze_mem(vector_scope, block2mem, block2firstmem);

#ifdef DUMP_VECTORIZER
                        std::cerr << "PreAll\n";
                        DUMP_BLOCK(latch);
#endif

                        //Step 1: Move along the schedule, add mem param to all split nodes and all join nodes
                        for (auto split_old : encloses_splits[latch_old]) {
                            Continuation* split_cont = const_cast<Continuation*>(split_old->as<Continuation>());

                            Continuation * then_new = const_cast<Continuation*>(def2def_[split_cont->arg(1)]->as<Continuation>());
                            Continuation * else_new = const_cast<Continuation*>(def2def_[split_cont->arg(2)]->as<Continuation>());
                            
                            Continuation_MemParam(then_new, vector_scope, block2mem, block2firstmem);
                            Continuation_MemParam(else_new, vector_scope, block2mem, block2firstmem);
                        }

                        {//Also extend everything related to the top-level latch.
                            Continuation* split_cont = const_cast<Continuation*>(latch_old->as<Continuation>());

                            Continuation * then_new = const_cast<Continuation*>(def2def_[split_cont->arg(1)]->as<Continuation>());
                            Continuation * else_new = const_cast<Continuation*>(def2def_[split_cont->arg(2)]->as<Continuation>());

                            Continuation_MemParam(then_new, vector_scope, block2mem, block2firstmem);
                            Continuation_MemParam(else_new, vector_scope, block2mem, block2firstmem);
                        }

                        //Check that all join nodes already take a mem parameter.
                        for (auto join_old : joins_old) {
                            auto join_new = def2def_[join_old]->as<Continuation>();
                            assert(is_mem(join_new->param(0)));
                        }

                        //Warning: There is a good chance that the mem-monad will be regarded as being broken from this point on, until the latch has its predicated continuation added.

                        //TODO: I might need to add mem to the br_vec intrinsic very early.
                        
                        GIDMap<const Continuation*, Array<const Def *>> join_caches(joins_old.size());
                        size_t num_enclosed_splits = encloses_splits[latch_old].size();
                        assert(num_enclosed_splits <= 1 && "There should only  be one enclosed split right now: then_new");
                        GIDMap<const Continuation*, const Def *> predicate_caches(num_enclosed_splits ? num_enclosed_splits : 1); //Size must not be 0.

                        { //mem scope
                            const Def *mem = block2mem[latch];
                            assert(mem);

                            for (auto join_old : joins_old) {
                                auto join = def2def_[join_old]->as<Continuation>();
                                size_t cache_size = join->num_params() - 1;
                                join_caches[join] = Array<const Def *>(cache_size);

#ifndef OLDALLOC
                                if (cache_size && !current_frame) {
                                    auto mem = vectorized->mem_param();
                                    auto enter = world_.enter(mem)->as<Enter>();
                                    current_frame = enter->out_frame();
                                    auto newmem = enter->out_mem();
                                    for (auto use : mem->copy_uses()) {
                                        auto use_inst = use.def();
                                        if (use_inst == enter)
                                            continue;
                                        assert(use_inst);
                                        int index = use.index();
                                        Def* olduse = const_cast<Def*>(use_inst);
                                        olduse->unset_op(index);
                                        olduse->set_op(index, newmem);
                                    }
                                    if (block2mem[vectorized] == mem)
                                        block2mem[vectorized] = newmem;
                                }
#endif

                                for (size_t i = 0; i < cache_size; i++) {
#ifdef OLDALLOC
                                    auto t = world_.alloc(join->param(i + 1)->type(), mem, Debug("join_cache_branch"));
                                    mem = world_.extract(t, (int) 0);
                                    join_caches[join][i] = world_.extract(t, 1);
#else
                                    assert(current_frame);
                                    auto t = world_.slot(join->param(i + 1)->type(), current_frame, Debug("join_cache_branch"));
                                    join_caches[join][i] = t;
#endif
                                }
                            }

                            if (num_enclosed_splits) {
                                if (!current_frame) {
                                    auto mem = vectorized->mem_param();
                                    auto enter = world_.enter(mem)->as<Enter>();
                                    current_frame = enter->out_frame();
                                    auto newmem = enter->out_mem();
                                    for (auto use : mem->copy_uses()) {
                                        auto use_inst = use.def();
                                        if (use_inst == enter)
                                            continue;
                                        assert(use_inst);
                                        int index = use.index();
                                        Def* olduse = const_cast<Def*>(use_inst);
                                        olduse->unset_op(index);
                                        olduse->set_op(index, newmem);
                                    }
                                    if (block2mem[vectorized] == mem)
                                        block2mem[vectorized] = newmem;
                                }

                                auto false_elem = world_.zero(world_.type_bool());
                                Array<const Def *> elements(vector_width);
                                for (size_t i = 0; i < vector_width; i++)
                                    elements[i] = false_elem;
                                auto zero_predicate = world_.vector(elements);
                                for (auto split_old : encloses_splits[latch_old]) {
                                    const Continuation* split_new = def2def_[split_old]->as<Continuation>();
                                    assert(split_new);
#ifdef OLDALLOC
                                    auto t = world_.alloc(zero_predicate->type(), mem, Debug("predicate_cache"));
                                    mem = world_.extract(t, (int) 0);
                                    const Def * pred_cache = world_.extract(t, 1);
#else
                                    assert(current_frame);
                                    auto pred_cache = world_.slot(zero_predicate->type(), current_frame, Debug("predicate_cache"));
#endif
                                    mem = world_.store(mem, pred_cache, zero_predicate);
                                    predicate_caches[split_new] = pred_cache;
                                }
                            }

                            //TODO: This new mem should replace all old mem instances!

                            block2mem[latch] = mem;
                        }

                        GIDMap<const Continuation*, const Continuation *> pre_joins(joins_old.size());
                        for (auto join_old : joins_old) {
                            //join == some_latch can occur at times!

                            auto join = def2def_[join_old]->as<Continuation>();
                            Continuation * pre_join = world_.continuation(world_.fn_type({world_.mem_type()}), Debug("branch_merge"));

                            const Def* mem = pre_join->param(0);
                            Array<const Def*> join_params(join->num_params());
                            auto &join_cache = join_caches[join];

                            for (size_t i = 1; i < join->num_params(); i++) {
                                auto load = world_.load(mem, join_cache[i - 1]);
                                auto value = world_.extract(load, (int) 1);
                                mem = world_.extract(load, (int) 0);
                                join_params[i] = value;
                            }
                            join_params[0] = mem;
                            pre_join->jump(join, join_params, latch->debug());

                            pre_joins[join] = pre_join;
                        }

                        //TODO: This code still feels incomplete and broken. Devise a better way to handle this rewiring of the mem-monad.
                        //Maybe I should extend all cases surrounding a latch with a memory operand with code similar to schedule verification.
                        //I depend on the schedule from time to time. This is not good.
                        Array<const Continuation*> splits(encloses_splits[latch_old].size()+1);
                        splits[0] = latch_old;
                        size_t i = 1;
                        for (auto split_old : encloses_splits[latch_old])
                            splits[i++] = split_old;
                        std::sort (splits.begin(), splits.end(), [&](const Continuation *ac, const Continuation *bc) {
                                Continuation *a = const_cast<Continuation*>(ac);
                                Continuation *b = const_cast<Continuation*>(bc);
                                return div_analysis_->dominatedBy[a].contains(b);
                            });

                        auto vec_mask = widen(world_.type_bool());

                        GIDMap<const Continuation*, const Continuation*> rewired_predecessors;

#ifdef DUMP_VECTORIZER
                        std::cerr << "Before rewiring\n";
                        DUMP_BLOCK(vectorized);
#endif

                        for (size_t current_split_index = 0; current_split_index < splits.size(); current_split_index++) {
                            auto split_old = splits[current_split_index];
                            auto split = const_cast<Continuation*>(def2def_[split_old]->as<Continuation>());
                            assert(split);

                            const Continuation * then_old = split_old->arg(1)->as<Continuation>();
                            Continuation * then_new = const_cast<Continuation*>(def2def_[then_old]->as<Continuation>());
                            assert(then_new);

                            const Continuation * else_old = split_old->arg(2)->as<Continuation>();
                            Continuation * else_new = const_cast<Continuation*>(def2def_[else_old]->as<Continuation>());
                            assert(else_new);


                            //TODO: These predicates need to be extended to include the cached predicates as well.
                            //TODO: We need to store predicates as well!
                            const Def* predicate_true = split->arg(0);

                            assert(predicate_true);
                            if (!predicate_true->type()->isa<VectorType>() || !predicate_true->type()->as<VectorType>()->is_vector()) {
                                Array<const Def *> elements(vector_width);
                                for (size_t i = 0; i < vector_width; i++) {
                                    elements[i] = predicate_true;
                                }
                                predicate_true = world_.vector(elements);
                            }
                            assert(predicate_true->type()->isa<VectorType>() && predicate_true->type()->as<VectorType>()->is_vector());

                            const Def* predicate_false = world_.arithop_not(predicate_true);
                            assert(predicate_false);

                            Continuation * then_new_back = world_.continuation(world_.fn_type({world_.mem_type()}), Debug("branch_true_back"));
                            Continuation * else_new_back = world_.continuation(world_.fn_type({world_.mem_type()}), Debug("branch_false_back"));

                            auto new_jump_latch = world_.predicated(vec_mask);

                            { //mem scope
                                const Def* mem = block2mem[split];
                                assert(mem);

                                assert(then_new != then_new_back);
                                split->jump(new_jump_latch, { mem, predicate_true, then_new, then_new_back }, latch->debug());
                            }

                            //Connect then-nodes to then-back
                            //If there are loops present on the then-side, we will not find an appropriate join node for that and the then_back node will be executed once all vector elements are done executing the loop.
                            //TODO: There might be an issue: There might be loops with the exit being the "then" case, then and else should be switched then!
                            for (auto join_old : joins_old) {
                                auto join = const_cast<Continuation*>(def2def_[join_old]->as<Continuation>());
                                assert(join);
                                auto &join_cache = join_caches[join];

                                for (auto pred_old : join_old->preds()) {
                                    if (!div_analysis_->dominatedBy[pred_old].contains(const_cast<Continuation*>(then_old)) && pred_old != then_old)
                                        continue;
                                    auto pred = const_cast<Continuation*>(def2def_[pred_old]->as<Continuation>());
                                    if (rewired_predecessors.contains(pred))
                                        if (rewired_predecessors[pred] == nullptr)
                                            continue;

                                    //get old mem parameter, if possible.
                                    assert(pred->arg(0)->type()->isa<MemType>());
                                    const Def* mem;
                                    mem = block2mem[pred];
                                    if (!mem)
                                        mem = pred->arg(0);
                                    assert(mem);

                                    for (size_t j = 0; j < pred->num_args(); j++) {
                                        if (j == 0)
                                            continue;
                                        assert(join_cache[j - 1]);
                                        mem = world_.store(mem, join_cache[j - 1], pred->arg(j));
                                    }

                                    pred->jump(then_new_back, { mem });
                                    rewired_predecessors.emplace(pred, nullptr);

                                    block2mem[pred] = mem;
                                }
                            }


                            //connect then-back to else-branch
                            auto new_jump_then = world_.predicated(vec_mask);
                            assert(else_new != else_new_back);
                            then_new_back->jump(new_jump_then, { then_new_back->mem_param(), predicate_false, else_new, else_new_back }, latch->debug());

                            Continuation *else_join_cache = nullptr;

                            //Connect else-nodes to else-back
                            for (auto join_old : joins_old) {
                                auto join = const_cast<Continuation*>(def2def_[join_old]->as<Continuation>());
                                assert(join);
                                auto &join_cache = join_caches[join];

                                for (auto pred_old : join_old->preds()) {
                                    if (!div_analysis_->dominatedBy[pred_old].contains(const_cast<Continuation*>(else_old)) && pred_old != else_old)
                                        continue;
                                    auto pred = const_cast<Continuation*>(def2def_[pred_old]->as<Continuation>());
                                    if (rewired_predecessors.contains(pred))
                                        if (rewired_predecessors[pred] == nullptr)
                                            continue;

                                    assert(!else_join_cache);
                                    else_join_cache = join;

                                    //get old mem parameter, if possible.
                                    assert(pred->arg(0)->type()->isa<MemType>());
                                    const Def* mem;
                                    mem = block2mem[pred];
                                    if (!mem)
                                        mem = pred->arg(0);
                                    assert(mem);

                                    for (size_t j = 0; j < pred->num_args(); j++) {
                                        if (j == 0)
                                            continue;
                                        assert(join_cache[j - 1]);
                                        mem = world_.store(mem, join_cache[j - 1], pred->arg(j));
                                    }

                                    auto pre_join = pre_joins[join];
                                    assert(pre_join);

                                    pred->jump(else_new_back, { mem });
                                    rewired_predecessors.emplace(pred, pre_join);

                                    block2mem[pred] = mem;
                                }
                            }

                            auto true_elem = world_.one(world_.type_bool());
                            Array<const Def *> elements(vector_width);
                            for (size_t i = 0; i < vector_width; i++) {
                                elements[i] = true_elem;
                            }
                            auto one_predicate = world_.vector(elements);

                            auto new_jump_else = world_.predicated(vec_mask);

                            if (!else_join_cache) {
                                //There is no join after else. This should only occur with loops, and else_back should not be reachable in this case.
                                //TODO: This should also imply that the current mask is no fully populated, as the else-block is only reachable once all loop instances are done executing.
                                
                                //else_new_back->jump(else_new->op(0), { else_new_back->mem_param() });
                                else_new_back->jump(new_jump_else, { else_new_back->mem_param(), one_predicate, else_new->op(0), else_new->op(0)}, latch->debug()); //TODO: This is only correct if the target is a return.
                            } else {
                                //connect else-back to the cached join.
                                
                                auto pre_join = pre_joins[else_join_cache];
                                assert(pre_join);
                                else_new_back->jump(new_jump_else, { else_new_back->mem_param(), one_predicate, pre_join, pre_join }, latch->debug());
                            }

                            //std::cerr << "After rewiring " << current_split_index + 1 << "\n";
                            //rewired_predecessors.dump();
                            //Scope(vectorized).dump();
                        }

                        //world_.dump();

                        /*for (auto join_old : joins_old) {
                            //join == some_latch can occur at times!
                            auto join = def2def_[join_old]->as<Continuation>();

                            for (auto use : join->uses()) {
                                if(use != pre_joins[join]) {
                                    std::cerr << "Found problem\n";
                                    Scope(vectorized).dump();
                                    join_old->dump();
                                    join->dump();
                                    use->dump();
                                    assert(false);
                                }
                            }
                        }*/
                    }

#ifdef DUMP_VECTORIZER
                    DUMP_BLOCK(vectorized);
#endif
                }

                }

                delete div_analysis_;

    //Task 4: Rewrite vectorize call
                if (vectorized) {
                    const Continuation* vectorized_flat = flatten_continuation(vectorized, world_);
                    assert(vectorized_flat);

                    for (auto caller : cont->preds()) {
                        Array<const Def*> args(vectorized_flat->num_params());

                        args[0] = caller->arg(0); //mem
                        //args[1] = caller->arg(1); //width
                        Array<const Def*> defs(vector_width);
                        for (size_t i = 0; i < vector_width; i++) {
                            defs[i] = world_.literal_qs32(i, caller->arg(1)->debug_history());
                        }
                        args[1] = world_.vector(defs, caller->arg(1)->debug_history());

                        for (size_t p = 2; p < vectorized_flat->num_params(); p++) {
                            args[p] = caller->arg(p + 1);
                        }

                        caller->jump(vectorized_flat, args, caller->debug());
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
        //llvm::TimeRegion clean_time(time_clean);
        //world_.cleanup();
    }
#ifdef DUMP_VECTORIZER
    std::cout << "Post Cleanup\n";
    world_.dump();
#endif

    return false;
}

bool vectorize(World& world) {
    world.VLOG("start vectorizer");
    //world.dump();
    auto res = Vectorizer(world).run();

    //world.dump();

    //flatten_vectors(world);
    //world.cleanup();

    world.dump();
    debug_verify(world);

    world.VLOG("end vectorizer");
    return res;
}

}
