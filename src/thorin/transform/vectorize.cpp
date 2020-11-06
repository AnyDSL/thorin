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
    World& world_;
    size_t boundary_;
    ContinuationSet done_;
    ContinuationMap<bool> top_level_;

    std::queue<Continuation*> queue_;
    void enqueue(Continuation* continuation) {
        if (continuation->gid() < 2 * boundary_ && done_.emplace(continuation).second)
            queue_.push(continuation);
    }

    void divergence_analysis(Continuation *continuation);

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

#if 0
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

        //std::cerr << "\n";
        //cont->dump();

        auto mydom = dominatedBy[cont];

        for (auto succ : cont->succs()) {
            if (succ->is_intrinsic())
                continue;
            //succ->dump();
            if (mydom.contains(succ)) {
                //std::cerr << "Loop registered\n";
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
            //cont->dump();
        }

        for (auto succ : cont->succs())
            if (done.emplace(succ).second)
                queue.push(succ);
    }

    //std::cerr << "Chapter 5\n";

    //Step 2.2: Chapter 5, alg. 1: Construct labelmaps.
    for (auto *split : splitNodes) {
        done.clear();

        GIDMap<Continuation*, Continuation*> LabelMap;

        //std::cerr << "\nSplit analysis\n";
        //split->dump();

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
            //std::cerr << "Predecessors\n";
            //cont->dump();
            //keys.dump();

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
                //std::cerr << "Successors\n";
                //cont->dump();
                //successors(cont).dump();
                for (auto succ : successors(cont))
                    queue.push(succ);
            }
        }

        relJoins[split] = Joins;

        //std::cerr << "Split node\n";
        //split->dump();
        //std::cerr << "Labelmap:\n";
        //LabelMap.dump();
        //std::cerr << "Joins:\n";
        //Joins.dump();
        //std::cerr << "End\n";
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
            std::cerr << "Varying values in\n";
            scope.dump();
            for (auto def : scope.defs()) {
                uniform[def] = Varying;
                def_queue.push(def);
            }
        }
    }

    for (auto it : relJoins) {
        it.first->dump();
        it.second.dump();
    }

    std::cerr << "\n";

    //Step 4.3: Profit?
    while (!def_queue.empty()) {
        const Def *def = pop(def_queue);

        std::cerr << "Will analyze ";
        def->dump();

        if (uniform[def] == Uniform)
            continue;

        for (auto use : def->uses()) {
            auto old_state = uniform[use];

            if (old_state == Uniform) {
                //Communicate Uniformity over continuation parameters
                Continuation *cont = use.def()->isa_continuation();
                if (cont) {
                    cont->dump();
                    bool is_op = false; //TODO: this is not a good filter for finding continuation calls!
                    int opnum = 0;
                    for (auto param : cont->ops()) {
                        if (param == def) {
                            is_op = true;
                            break;
                        }
                        opnum++;
                    }
                    std::cerr << is_op << "\n";
                    std::cerr << opnum << "\n";
                    auto target = cont->ops()[0]->isa_continuation();
                    if (is_op && target && target->is_intrinsic() && opnum == 1 && relJoins.find(cont) != relJoins.end()) {
                        ContinuationSet joins = relJoins[cont];
                        for (auto join : joins) {
                            Scope scope(join);
                            std::cerr << "Varying values in\n";
                            scope.dump();
                            for (auto def : scope.defs()) { //TODO: only parameters are verying, not the entire continuation!
                                std::cerr << "Push def ";
                                def->dump();
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
                                std::cerr << "Push param ";
                                target_param->dump();
                            }
                        }
                    } else {
                        std::cerr << "\nNot Found\n";
                        cont->dump();
                        for (auto it : relJoins) {
                            it.first->dump();
                            it.second.dump();
                        }
                        std::cerr << "\n";
                    }
                } else {
                    uniform[use] = Varying;
                    def_queue.push(use);
                    std::cerr << "Push ";
                    use->dump();
                }
            }
        }
    }

    std::cerr << "\n";

    for (auto uni : uniform) {
        if (uni.second == Varying)
            std::cerr << "Varying ";
        if (uni.second == Uniform)
            std::cerr << "Uniform ";
        uni.first->dump();
    }

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
            cont->dump_head();

            for (auto pred : cont->preds()) {
                auto *kernarg = dynamic_cast<const Global *>(pred->arg(2));
                assert(kernarg && "Not sure if all kernels are always declared globally");
                assert(!kernarg->is_mutable() && "Self transforming code is not supported here!");
                auto *kerndef = kernarg->init()->isa_continuation();
                assert(kerndef && "We need a continuation for vectorization");

    //Task 1.2: Divergence Analysis for each vectorize block
    //Warning: Will fail to produce meaningful results or rightout break the program if kerndef does not dominate its subprogram
                DivergenceAnalysis(kerndef).run();

            }
            std::cerr << "Continuation end\n\n";
        }
        //if cont is vectorize:
        //find all calls, repeat the rest of this for all of them with the respective set of continuations being used.
        
        
        for (auto succ : cont->succs())
            enqueue(succ);
    }


    return false;

    //Task 2: Communicate, annotate the IR, or transform it alltogether

    //Task 3: Widening (during CodeGen)
    //TODO: Uniform branches might still need masking. => Predicate generation relevant!

    //world.cleanup();
}

bool vectorize(World& world) {
    VLOG("start vectorizer");
    auto res = Vectorizer(world).run();
    VLOG("end vectorizer");
    return res;
}

}
