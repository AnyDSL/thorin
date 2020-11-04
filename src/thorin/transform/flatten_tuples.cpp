#include "thorin/lam.h"
#include "thorin/world.h"
#include "thorin/transform/mangle.h"
#include "thorin/analyses/verify.h"
#include "thorin/util/log.h"

namespace thorin {

class FlattenTuples {
    World& world_;
    Type2Type flat_types_;
    Def2Def flat_defs_;

    const Type* flatten_type(const Type* type) {
        if (auto flat_type = find(flat_types_, type))
            return flat_type;
        const Type* flat_type = nullptr;
        if (type->is_nominal())
            flat_types_[type] = flat_type = type;

        if (type->isa<TupleType>()) {
            std::vector<const Type*> flat_ops;
            for (auto op : type->ops()) {
                auto flat_op = flatten_type(op);
                if (flat_op->isa<TupleType>())
                    flat_ops.insert(flat_ops.end(), flat_op->ops().begin(), flat_op->ops().end());
                else
                    flat_ops.push_back(flat_op);
            }
            flat_type = world_.tuple_type(flat_ops);
        } else {
            Array<const Type*> flat_ops(type->num_ops());
            for (size_t i = 0, n = type->num_ops(); i < n; ++i)
                flat_ops[i] = flatten_type(type->op(i));
            if (!flat_type)
                flat_type = type->rebuild(flat_ops);
            else {
                for (size_t i = 0, n = type->num_ops(); i < n; ++i)
                    flat_type->as<NominalType>()->set(i, flat_ops[i]);
            }
        }
        return flat_types_[type] = flat_type;
    }

    size_t get_insert_or_extract_index(const Type* old_type, size_t old_index) {
        // This computes the insertion/extraction index into the flattened type,
        // given the old, unflatten type and the corresponding index.
        assert(old_type->isa<TupleType>());
        size_t index = 0;
        for (size_t i = 0, n = std::min(old_type->num_ops(), old_index); i < n; ++i) {
            auto flat_op = flatten_type(old_type->op(i));
            if (flat_op->isa<TupleType>())
                index += flat_op->num_ops();
            else
                index++;
        }
        return index;
    }

    const Def* flatten_def(const Def* def) {
        if (auto flat_def = find(flat_defs_, def))
            return flat_def;
        const Def* flat_def = nullptr;
        if (def->is_nominal()) {
            flat_def = def;
            if (flatten_type(def->type()) != def->type())
                flat_def = def->stub(flatten_type(def->type()));
            flat_defs_[def] = flat_def;
        }

        if (def->isa<Tuple>()) {
            std::vector<const Def*> flat_ops;
            for (auto op : def->ops()) {
                auto flat_op = flatten_def(op);
                if (flat_op->type()->isa<TupleType>()) {
                    for (size_t i = 0, n = flat_op->type()->num_ops(); i < n; ++i)
                        flat_ops.push_back(world_.extract(flat_op, i));
                } else
                    flat_ops.push_back(flat_op);
            }
            flat_def = world_.tuple(flat_ops);
        } else if (def->isa<Insert>() && def->op(0)->type()->isa<TupleType>()) {
            auto flat_agg = flatten_def(def->op(0));
            auto flat_val = flatten_def(def->op(2));
            auto index = get_insert_or_extract_index(def->op(0)->type(), primlit_value<u64>(def->op(1)));
            if (flat_val->type()->isa<TupleType>()) {
                // If the inserted value is a tuple, we need to split this insert
                // into one insert for each element of the inserted tuple.
                for (size_t i = 0, n = flat_val->type()->num_ops(); i < n; ++i)
                    flat_agg = world_.insert(flat_agg, index + i, world_.extract(flat_val, i));
                flat_def = flat_agg;
            } else {
                flat_def = world_.insert(flat_agg, index, flat_val);
            }
        } else if (def->isa<Extract>() && def->op(0)->type()->isa<TupleType>()) {
            auto flat_agg = flatten_def(def->op(0));
            auto index = get_insert_or_extract_index(def->op(0)->type(), primlit_value<u64>(def->op(1)));
            if (def->type()->isa<TupleType>()) {
                // If the result of the extract is a tuple, we need to split this extract
                // into several ones, and then build a tuple out of each.
                auto last_index = get_insert_or_extract_index(def->op(0)->type(), primlit_value<u64>(def->op(1)) + 1);
                Array<const Def*> flat_ops(last_index - index);
                for (size_t i = 0, n = flat_ops.size(); i < n; ++i)
                    flat_ops[i] = world_.extract(flat_agg, index + i);
                flat_def = world_.tuple(flat_ops);
            } else {
                flat_def = world_.extract(flat_agg, index);
            }
        } else {
            Array<const Def*> flat_ops(def->num_ops());
            for (size_t i = 0, n = def->num_ops(); i < n; ++i)
                flat_ops[i] = flatten_def(def->op(i));
            if (!flat_def)
                flat_def = def->rebuild(flat_ops);
            else {
                for (size_t i = 0, n = def->num_ops(); i < n; ++i)
                    flat_def->as_lam()->update_op(i, flat_ops[i]);
                // Mark rewritten lambdas as internal so as to remove them during cleanup
                if (flat_def != def)
                    def->as_lam()->make_internal();
            }
        }
        return flat_defs_[def] = flat_def;
    }

public:
    FlattenTuples(World& world) : world_(world) {}

    void run() {
        for (auto lam : world_.exported_lams())
            flatten_def(lam);
    }
};

void flatten_tuples(World& world) {
    FlattenTuples(world).run();
    world.cleanup();
}

}
