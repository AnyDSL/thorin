#include "thorin/analyses/divergence.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/looptree.h"
#include "thorin/analyses/domtree.h"

//#define DUMP_DIV_ANALYSIS
//#define DUMP_LOOP_ANALYSIS

namespace thorin {

class DivergenceRecStreamer {
public:
    DivergenceRecStreamer(Stream& s
            , DivergenceAnalysis* div_analysis
            , size_t max)
        : s(s)
        , div_analysis(div_analysis)
        , max(max)
    {}

    void run();
    void run(const Def*);

    Stream& s;
    DivergenceAnalysis* div_analysis;
    size_t max;
    unique_queue<ContinuationSet> conts;
    DefSet defs;
};


void DivergenceRecStreamer::run (const Def* def) {
    if (def->no_dep() || !defs.emplace(def).second) return;

    for (auto op : def->ops()) { // for now, don't include debug info and type
        if (auto cont = op->isa_continuation()) {
            if (max != 0) {
                if (conts.push(cont)) --max;
            }
        } else {
            run(op);
        }
    }

    if (auto cont = def->isa_continuation())
        s.fmt("{}: {} = {}({, })", cont, cont->type(), cont->callee(), cont->args())
        .fmt(" < {} >", (div_analysis->getUniform(cont->callee()) == DivergenceAnalysis::State::Uniform)  ? "Uni" : "Div");
    else if (!def->no_dep() && !def->isa<Param>())
        def->stream1(s.fmt("{}: {} = ", def, def->type()))
        .fmt(" < {} >", (div_analysis->getUniform(def) == DivergenceAnalysis::State::Uniform)  ? "Uni" : "Div")
        .endl();
}


void DivergenceRecStreamer::run() {
    while (!conts.empty()) {
        auto cont = conts.pop();
        s.endl().endl();

        if (!cont->empty()) {
            s.fmt("{}: {} = {{ ", cont->unique_name(), cont->type());
            for (size_t i = 0; i < cont->num_params(); i++) {
                auto param = cont->param(i);
                s.fmt("[{}:{}]", param, (div_analysis->getUniform(param) == DivergenceAnalysis::State::Uniform)  ? "Uni" : "Div");
            }
            s.fmt("[pred:{}]", div_analysis->isPredicated[cont] ? "pred" : "no_pred");
            s.fmt("\t\n");
            run(cont);
            s.fmt("\b\n}}");
        } else {
            s.fmt("{}: {} = {{ <unset> }}", cont->unique_name(), cont->type());
        }
    }
}


void DivergenceAnalysis::dump () {
    Stream s(std::cout);
    DivergenceRecStreamer rec(s, this, std::numeric_limits<size_t>::max());
    for (auto& block : schedule(Scope(const_cast<Continuation*>(base)))) {
        rec.conts.push(block);
        rec.run();
    }
    s.endl();
}


DivergenceAnalysis::State DivergenceAnalysis::getUniform(const Def * def) {
    return uniform[def];
}


ContinuationSet DivergenceAnalysis::successors(Continuation * cont) {
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


ContinuationSet DivergenceAnalysis::predecessors(Continuation * cont) {
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


void DivergenceAnalysis::computeLoops() {
    Scope scope = Scope(base);
    auto& cfg = scope.f_cfg();
    const LoopTree<true> &loops = cfg.looptree();
    const DomTreeBase<true> &domtree = cfg.domtree();

    auto& bcfg = scope.b_cfg();
    const DomTreeBase<false> &bdomtree = bcfg.domtree();

    std::queue <Continuation*> queue;
    ContinuationSet done;
    //Step 1: construct head rewired CFG. (should probably be implicit)
    //Step 1.1: Find Loops

    //Step 1.1.1: Dominance analysis caching
    for (auto cfnode : cfg.reverse_post_order()) {
        auto node = cfnode->continuation();

        ContinuationSet mydom;
        mydom.emplace(node);

        auto idom = cfnode;
        while (idom != cfg.entry()) {
            idom = domtree.idom(idom);
            mydom.emplace(idom->continuation());
        }

        dominatedBy[node] = mydom;
    }

    for (auto cfnode : bcfg.reverse_post_order()) {
        auto node = cfnode->continuation();

        ContinuationSet mydom;
        mydom.emplace(node);

        auto idom = cfnode;
        while (idom != bcfg.entry()) {
            idom = bdomtree.idom(idom);
            mydom.emplace(idom->continuation());
        }

        sinkedBy[node] = mydom;
    }

#ifdef DUMP_LOOP_ANALYSIS
    base->dump();

    for (auto elem : dominatedBy) {
        std::cerr << "dominated by\n";
        elem.first->dump();
        for (auto elem2 : elem.second)
            elem2->dump();
        std::cerr << "end\n";
    }
    std::cerr << "\n";
#endif

    //Step 1.1.2: Find Loop Exit Nodes
    for (auto cfnode : cfg.reverse_post_order()) {
        auto leaf = loops[cfnode];
        auto parent = leaf->parent();
        while (parent) {
            for  (auto latch : parent->cf_nodes())
                loopBodies[latch->continuation()].emplace(cfnode->continuation());
            parent = parent->parent();
        }
    }

    for (auto it : loopBodies) {
        auto header = it.first;
        auto body = it.second;

        for (auto elem : body)
            for (auto succ : elem->succs())
                if (!succ->is_intrinsic() && !succ->is_imported() && !body.contains(succ))
                    loopExits[header].emplace(succ);
    }
}


void DivergenceAnalysis::run() {
    computeLoops();

#ifdef DUMP_DIV_ANALYSIS
    DUMP_BLOCK(base);
    std::cerr << "Loops are\n";
    for (auto elem : loopBodies) {
        auto exits = loopExits[elem.first];
        std::cerr << "Header\n";
        elem.first->dump();
        std::cerr << "Body\n";
        for (auto elem : elem.second)
            elem->dump();
        std::cerr << "Exits\n";
        ContinuationSet reachings;
        for (auto elem : exits) {
            elem->dump();
            for (auto suc : elem->succs())
                reachings.emplace(suc);
        }
        std::cerr << "Reachings\n";
        for (auto elem : reachings) {
            elem->dump();
        }
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
                for (auto succ : successors(cont)) {
                    if (succ->is_intrinsic() ||  succ->is_imported())
                        continue;
                    queue.push(succ);
                }
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
    //Step <removed>: Definite Reaching Definitions Analysis (see Chapter 6) (not at first, focus on CFG analysis first)
    //Note: I am currently not sure which is better, first doing the Definite Reachign Analysis, or first doing the CFG analysis.
    //Although needed in the paper, we do not require this. Code in Thorin is in SSA form.

    //Step 3: Vaule Uniformity
    //Step 3.1: Mark all Values as being uniform.
    Scope base_scope(base);
    for (auto def : base_scope.defs())
        uniform[def] = Uniform;

    std::queue <const Def*> def_queue;

    //Step 3.2: Mark varying defs
    //TODO: Memory Analysis: We need to track values that lie in memory slots!
    //This might not be so significant, stuff resides in memory for a reason.

    //Step 3.2.1: Mark incomming trip counter as varying.
    auto def = base->params_as_defs()[1];
    uniform[def] = Varying;
    def_queue.push(def);

    //Step 3.2.2: Mark everything in relevant joins as varying (for now at least)
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
    std::cerr << "\n";
    std::cerr << "RelJoins\n";
    for (auto it : relJoins) {
        std::cerr << "Head\n";
        it.first->dump();
        std::cerr << "Joins\n";
        for (auto s : it.second)
            s->dump();
    }
    std::cerr << "SplitParrents\n";
    for (auto it : splitParrents) {
        std::cerr << "Parent\n";
        it.first->dump();
        std::cerr << "Childs\n";
        for (auto s : it.second)
            s->dump();
    }

    std::cerr << "end\n";
    std::cerr << "\n";
#endif

    //Step 3.3: Profit?
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
#ifdef DUMP_DIV_ANALYSIS
                    if (target)
                        target->dump();
#endif
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
                        if (target->is_imported() || (target->is_intrinsic() && target->intrinsic() != Intrinsic::Match && target->intrinsic() != Intrinsic::Branch)) {
                            auto actualtarget = const_cast<Continuation*>(cont->op(cont->num_ops() - 1)->isa<Continuation>());
                            assert(actualtarget);
#ifdef DUMP_DIV_ANALYSIS
                            actualtarget->dump();
#endif
                            Debug de = target->debug();
                            if ((de.name.rfind("rv_ballot", 0) == 0) || (de.name.rfind("rv_any", 0) == 0)) {
                                for (auto param : actualtarget->params()) {
#ifdef DUMP_DIV_ANALYSIS
                                    std::cerr << "Mark rv intrinsic param uniform\n";
                                    param->dump();
#endif
                                    uniform[param] = Uniform;
                                }
                            } else  {
                                for (auto param : actualtarget->params()) {
#ifdef DUMP_DIV_ANALYSIS
                                    std::cerr << "Mark param varying\n";
                                    param->dump();
#endif
                                    uniform[param] = Varying;
                                    def_queue.push(param);
                                }
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

    //Step 4: Analyze predicate information for each continuation.
    //A continuation is considered predicated if it is part of the body of a conditional or loop.
    done.clear();
    assert(queue.empty());

    queue.push(base);
    done.emplace(base);

    while (!queue.empty()) {
        Continuation *cont = pop(queue);

        isPredicated[cont] = false;

        for (auto succ : cont->succs())
            if (done.emplace(succ).second)
                queue.push(succ);
    }

    for (auto join : relJoins) {
        done.clear();
        assert(queue.empty());

        for (auto succ : join.first->succs()) {
            if (join.second.contains(succ))
                continue;
            if (done.emplace(succ).second)
                queue.push(succ);
        }

        while (!queue.empty()) {
            Continuation *cont = pop(queue);

            isPredicated[cont] = true;

            for (auto succ : cont->succs()) {
                if (join.second.contains(succ))
                    continue;
                if (done.emplace(succ).second)
                    queue.push(succ);
            }
        }
    }

    for (auto loop : loopBodies) {
        for (auto cont : loop.second) {
            isPredicated[cont] = true;
        }
    }
    for (auto loop : loopExits) {
        for (auto cont : loop.second) {
            isPredicated[cont] = true;
        }
    }
}

}
