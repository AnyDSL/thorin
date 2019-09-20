#include "thorin/util.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/schedule.h"

namespace thorin {

class ResolveLoads {
public:
    ResolveLoads(World& world)
        : world_(world)
    {}

    bool resolve_loads() {
        todo_ = false;

        Scope::for_each(world_, [&] (Scope& scope) {
            resolve_loads(scope);
            scope.update(); // TODO only updated when actually needed
        });
        // Remove slots that only have stores
        for (auto& pair : safe_slots_) {
            if (!pair.second) continue;
            if (are_ptr_uses_safe(pair.first, false)) {
                replace_ptr_uses(pair.first);
                todo_ = true;
            }
        }
        return todo_;
    }

    void resolve_loads(const Scope& scope) {
        for (auto node : scope.f_cfg().reverse_post_order()) {
            auto nom = node->nominal();
            for (auto param : nom->params()) {
                if (param->type()->isa<Mem>()) {
                    Def2Def mapping;
                    resolve_loads(param, mapping);
                }
            }
        }
    }

    void resolve_loads(const Def* mem, Def2Def& mapping) {
        // Traverse the tree of memory objects and
        // incrementally build the contents of each
        // safe slot/immutable global
        while (mem) {
            // This loop iterates through all uses, and processes them recursively.
            // The last use is treated separately to be able to re-use the mapping.
            auto uses = mem->copy_uses();
            size_t i = 0, n = uses.size();
            for (auto it = uses.begin(); i < n; ++i, ++it) {
                if (i == n - 1) {
                    mem = process_use(*it, mapping);
                } else {
                    Def2Def split_mapping = mapping;
                    auto next_mem = process_use(*it, split_mapping);
                    resolve_loads(next_mem, split_mapping);
                }
            }
            if (n == 0)
                break;
        }
    }

    const Def* process_use(const Def* mem_use, Def2Def& mapping) {
        if (auto load = mem_use->isa<Load>()) {
            // Try to find the slot corresponding to this load
            if (auto slot = find_slot(load->ptr())) {
                // If the slot has been found and is safe, try to find a value for it
                auto slot_value = get_value(slot, mapping);
                auto load_value = extract_from_slot(load->ptr(), slot_value, load->debug());
                // If the loaded value is completely specified, replace the load
                if (!contains_top(load_value)) {
                    todo_ = true;
                    load->replace(world_.tuple({ load->mem(), load_value }));
                }
            }
            return load->out_mem();
        } else if (auto store = mem_use->isa<Store>()) {
            // Try to find the slot corresponding to this store
            if (auto slot = find_slot(store->ptr())) {
                // If the slot has been found and is safe, try to find a value for it
                auto slot_value = get_value(slot, mapping);
                auto stored_value = insert_to_slot(store->ptr(), slot_value, store->val(), store->debug());
                mapping[slot] = stored_value;
            }
            return store->out_mem();
        } else if (auto slot = mem_use->isa<Slot>()) {
                if (slot && is_safe_slot(slot))
                    mapping[slot] = world_.bot(slot->type()->as<Ptr>()->pointee());
            return slot->out_mem();
        } else {
            return nullptr;
        }
    }

    const Def* get_value(const Def* alloc, Def2Def& mapping) {
        auto it = mapping.find(alloc);
        if (it != mapping.end())
            return it->second;
        if (auto global = alloc->isa<Global>()) {
            // Immutable globals will remain set to their initial value
            if (!global->is_mutable())
                return mapping[alloc] = global->init();
        }
        // Nothing is known about this allocation yet
        return mapping[alloc] = world_.top(alloc->type()->as<Ptr>()->pointee(), alloc->debug());
    }

    const Def* extract_from_slot(const Def* ptr, const Def* slot_value, const Def* dbg) {
        while (auto bitcast = isa<Tag::Bitcast>(ptr))
            ptr = bitcast->arg();
        if (auto lea = isa<Tag::LEA>(ptr)) {
            auto [ptr, index] = lea->args<2>();
            return world_.extract_unsafe(extract_from_slot(ptr, slot_value, dbg), index, dbg);
        }
        return slot_value;
    }

    const Def* insert_to_slot(const Def* ptr, const Def* slot_value, const Def* insert_value, const Def* dbg) {
        std::vector<const Def*> indices;
        while (true) {
            if (auto bitcast = isa<Tag::Bitcast>(ptr)) {
                ptr = bitcast->arg();
            } else if (auto lea = isa<Tag::LEA>(ptr)) {
                auto [lea_ptr, index] = lea->args<2>();
                indices.push_back(index);
                ptr = lea_ptr;
            } else {
                break;
            }
        }
        size_t n = indices.size();
        if (n == 0)
            return insert_value;
        std::vector<const Def*> values(n + 1);
        values[n] = slot_value;
        values[0] = insert_value;
        for (size_t i = n - 1; i > 0; --i)
            values[i] = world_.extract(values[i + 1], indices[i], dbg);
        for (size_t i = 1; i <= n; ++i)
            values[i] = world_.insert(values[i], indices[i - 1], values[i - 1], dbg);
        return values[n];
    }

    bool contains_top(const Def* def) {
        if (is_top_.contains(def))
            return is_top_[def];
        if (def->isa<Top>()) {
            return is_top_[def] = true;
        } else if (!def->isa_nominal()) {
            for (auto op : def->ops()) {
                if (contains_top(op))
                    return is_top_[def] = true;
            }
            return is_top_[def] = false;
        } else {
            return is_top_[def] = false;
        }
    }

    bool is_safe_slot(const Def* slot) {
        assert(slot->isa<Slot>());
        if (safe_slots_.contains(slot)) return safe_slots_[slot];
        return safe_slots_[slot] = are_ptr_uses_safe(slot);
    }

    const Def* find_slot(const Def* ptr) {
        if (ptr->isa<Slot>() && is_safe_slot(ptr)) return ptr;
        if (ptr->isa<Global>() && !ptr->as<Global>()->is_mutable()) return ptr;
        if (auto lea = isa<Tag::LEA>(ptr)) return find_slot(lea->arg(0));
        if (auto bitcast = isa<Tag::Bitcast>(ptr)) return find_slot(bitcast->arg());
        return nullptr;
    }

    static void replace_ptr_uses(const Def* ptr) {
        for (auto& use : ptr->uses()) {
            if (auto store = use->isa<Store>()) {
                store->replace(store->mem());
            } else if (use->isa<Load>()) {
                assert(false);
            } else if (isa<Tag::LEA>(use) || isa<Tag::Bitcast>(use)) {
                replace_ptr_uses(use);
            } else {
                assert(false);
            }
        }
    }

    static bool are_ptr_uses_safe(const Def* ptr, bool allow_load = true) {
        for (auto& use : ptr->uses()) {
            if (use->isa<Store>()) {
                if (use.index() != 1) return false;
            } else if (isa<Tag::LEA>(use)) {
                if (!are_ptr_uses_safe(use.def(), allow_load)) return false;
            } else if (auto bitcast = isa<Tag::Bitcast>(use)) {
                // Support cast between pointers to definite and indefinite arrays
                auto ptr_to   = bitcast->type()->isa<Ptr>();
                auto ptr_from = bitcast->arg()->type()->isa<Ptr>();
                if (!ptr_to || !ptr_from)
                    return false;
                auto variadic_to   = ptr_to->pointee()->isa<Variadic>();
                auto variadic_from = ptr_from->pointee()->isa<Variadic>();
                if (!variadic_to || !variadic_from)
                    return false;
                if (variadic_to->body() != variadic_from->body())
                    return false;
                if (!are_ptr_uses_safe(use.def(), allow_load)) return false;
            } else if (!allow_load || !use->isa<Load>()) {
                return false;
            }
        }
        return true;
    }

private:
    bool todo_;
    World& world_;
    DefMap<bool> is_top_;
    DefMap<bool> safe_slots_;
};

bool resolve_loads(World& world) {
    return ResolveLoads(world).resolve_loads();
}

} // namespace thorin
