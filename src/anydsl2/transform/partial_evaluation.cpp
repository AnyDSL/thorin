#include <unordered_map>

#include "anydsl2/world.h"
#include "anydsl2/analyses/scope.h"
#include "anydsl2/be/air.h"
#include "anydsl2/analyses/schedule.h"
#include "anydsl2/analyses/looptree.h"
#include "anydsl2/transform/mangle.h"
#include "anydsl2/transform/merge_lambdas.h"

namespace anydsl2 {

#if 0
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
#endif

static size_t count(Lambda* dropped, size_t num) {
    if (dropped == nullptr)
        return num;

    Scope scope(dropped);
    assert(num >= scope.size());
    size_t result = num - scope.size();
    for (auto lambda : scope.rpo()) {
        if (lambda->to()->is_const()) {
            assert(lambda->to()->isa<Lambda>());
            ++result;
        }
    }

    return result;
}

void partial_evaluation(World& world) {
    //std::unordered_map<Call, Lambda*, CallHash, CallEqual> done;
    bool todo;

    //for (int counter = 0; counter < 1; ++counter) {
    do {
        todo = false;

        for (auto top : top_level_lambdas(world)) {
            Scope scope(top);
            for (auto lambda : scope.rpo()) {
                if (lambda->empty())
                    continue;
                if (auto to = lambda->to()->isa_lambda()) {
                    if (to->gid() == 74)
                        std::cout << "hey" << std::endl;
                    Scope scope(to);
                    GenericMap map;
                    bool res = to->type()->infer_with(map, lambda->arg_pi());
                    assert(res);
                    Lambda* e_dropped = nullptr;
                    Lambda* f_dropped = nullptr;
                    std::vector<Def> e_args, f_args;
                    std::vector<size_t> e_idx, f_idx;

                    for (size_t i = 0, e = lambda->num_args(); i != e; ++i) {
                        if (auto run = lambda->arg(i)->isa<Run>()) {
                            e_args.push_back(run);
                            e_idx.push_back(i);
                        }
                    }

                    if (!e_args.empty())
                        e_dropped = drop(scope, e_idx, e_args, map);
                    else
                        continue;

                    for (size_t i = 0, e = lambda->num_args(); i != e; ++i) {
                        auto arg = lambda->arg(i);
                        if (!arg->isa<Halt>()) {
                            if (arg->is_const())
                                arg = world.run(arg);
                            f_args.push_back(arg);
                            f_idx.push_back(i);
                        }
                    }

                    if (!f_args.empty())
                        f_dropped = drop(scope, f_idx, f_args, map);

                    // choose better variant
                    size_t num = scope.size();
                    size_t e_good = count(e_dropped, num);
                    size_t f_good = count(f_dropped, num);
                    bool use_f = f_good > e_good;
                    use_f = true;

                    std::vector<size_t>* idx = use_f ? &f_idx : &e_idx;
                    Lambda* dropped          = use_f ? f_dropped : e_dropped;

                    lambda->jump(dropped, lambda->args().cut(*idx));
                    todo = true;
                }
            }
        }
    }
    while (todo);
}

}
