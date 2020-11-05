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
    return Varying;
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
            cont->dump();
        }

        for (auto succ : cont->succs())
            if (done.emplace(succ).second)
                queue.push(succ);
    }

    std::cerr << "Chapter 5\n";

    done.clear();

    //Step 2.2: Chapter 5, alg. 1: Construct labelmaps.
    for (auto *split : splitNodes) {
        GIDMap<Continuation*, Continuation*> LabelMap;

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
            std::cerr << "Predecessors\n";
            cont->dump();
            keys.dump();

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
                std::cerr << "Successors\n";
                cont->dump();
                successors(cont).dump();
                for (auto succ : successors(cont))
                    queue.push(succ);
            }
        }

        relJoins[split] = Joins;

        std::cerr << "Split node\n";
        split->dump();
        std::cerr << "Labelmap:\n";
        LabelMap.dump();
        std::cerr << "Joins:\n";
        Joins.dump();
        std::cerr << "End\n";
    }

    //TODO: Heavy caching is of the essence.
    //Step 3: Definite Reaching Definitions Analysis (see Chapter 6) (not at first, focus on CFG analysis first)
    //Note: I am currently not sure which is better, first doing the Definite Reachign Analysis, or first doing the CFG analysis.

    done.clear();

    //Step 4: Vaule Uniformity
    //Step 4.1: Mark all Values as being uniform.
    Scope scope(base);
    for (auto def : scope.defs())
        uniform[def] = Uniform;

    //Step 4.2: Mark everything in relevant joins as varying (for now at least)
    for (auto it : relJoins) {
        Continuation *split = it.first;
        ContinuationSet joins = it.second;
        for (auto join : joins) {
            Scope scope(join);
            for (auto def : scope.defs())
                uniform[def] = Varying;
        }
    }

    //queue.push(base);
    //done.emplace(base);

    //std::cerr << "\n";
    //while (!queue.empty()) {
        //Continuation *cont = pop(queue);

        //std::cerr << "Node\n";
        //cont->dump();

        //scope.dump();


        //for (auto succ : cont->succs())
            //if (done.emplace(succ).second)
                //queue.push(succ);

        //std::cerr << "Node end\n";
    //}
    //std::cerr << "\n";

    for (auto uni : uniform) {
        if (uni.second == Varying)
            std::cerr << "Varying ";
        if (uni.second == Uniform)
            std::cerr << "Uniform ";
        uni.first->dump();
    }
    //Step 4.2: For all splits, and for all joins for those splits: Mark all merged values in join node as varying.
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

    //TODO: take a look at partial_evaluation for recomended controll flow.


    //Task 2: Communicate, annotate the IR, or transform it alltogether

    //Task 3: Widening (during CodeGen)

    //world.cleanup();
}

bool vectorize(World& world) {
    VLOG("start vectorizer");
    auto res = Vectorizer(world).run();
    VLOG("end vectorizer");
    return res;
}

}
