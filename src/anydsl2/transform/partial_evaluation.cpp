#include <unordered_map>

#include "anydsl2/world.h"
#include "anydsl2/analyses/scope.h"
#include "anydsl2/be/air.h"
#include "anydsl2/analyses/schedule.h"
#include "anydsl2/analyses/looptree.h"
#include "anydsl2/transform/mangle.h"
#include "anydsl2/transform/merge_lambdas.h"

namespace anydsl2 {

struct Call {
    Lambda* to;
    std::vector<const DefNode*> args;
    std::vector<size_t> idx;
};

struct CallHash { 
    size_t operator () (const Call& call) const { 
        return hash_combine(hash_combine(hash_value(call.to), 
                ArrayRef<const DefNode*>(call.args)), 
                ArrayRef<size_t>(call.idx));
    }
};

struct CallEqual { 
    bool operator () (const Call& call1, const Call& call2) const { 
        return call1.to == call2.to 
            && ArrayRef<const DefNode*>(call1.args) == ArrayRef<const DefNode*>(call2.args) 
            && ArrayRef<size_t>(call1.idx) == ArrayRef<size_t>(call2.idx);
    }
};

void partial_evaluation(World& world) {
    std::unordered_map<Call, Lambda*, CallHash, CallEqual> done;
    bool todo;

    do {
        todo = false;

        for (auto top : top_level_lambdas(world)) {
            Scope scope(top);
            for (auto lambda : scope.rpo()) {
                if (lambda->empty())
                    continue;
                if (auto to = lambda->to()->isa_lambda()) {
                    Call call;
                    call.to = to;

                    for (size_t i = 0, e = lambda->num_args(); i != e; ++i) {
                        if (auto run = lambda->arg(i)->isa<Run>()) {
                            call.args.push_back(run);
                            call.idx.push_back(i);
                        }
                    }

                    if (call.args.empty())
                        continue;

                    Lambda* dropped;
                    auto iter = done.find(call);
                    if (iter != done.end()) {
                        std::cout << "FOUND!!!" << std::endl;
                        dropped = iter->second;
                    } else {
                        GenericMap map;
                        bool res = to->type()->infer_with(map, lambda->arg_pi());
                        assert(res);
                        Array<Def> args(call.args.size());
                        for (size_t i = 0, e = args.size(); i != e; ++i)
                            args[i] = call.args[i];
                        dropped = drop(Scope(call.to), call.idx, args, map);
                        done[call] = dropped;
                        todo = true;
                    }

                    lambda->jump(dropped, lambda->args().cut(call.idx));
                }

            }
        }
        //std::cout << "---" << std::endl;
        //emit_air(world, false);
        //std::cout << "---" << std::endl;
    } while (todo);
    //std::cout << "FINAL:" << std::endl;
    //std::cout << "---------------------------" << std::endl;
    //emit_air(world, false);
}

}
