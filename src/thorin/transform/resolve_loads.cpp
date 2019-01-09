#include "thorin/analyses/scope.h"
#include "thorin/analyses/schedule.h"
#include "thorin/world.h"

namespace thorin {

class ResolveLoads {
public:
    ResolveLoads(World& world)
        : world_(world)
    {}

    bool resolve_loads() {
        todo_ = false;

        Scope::for_each(world_, [&] (const Scope& scope) {
            resolve_loads(scope);
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
            auto lam = node->lam();
            for (auto param : lam->params()) {
                if (param->type()->isa<MemType>()) {
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
            auto slot = find_slot(load->ptr());
            if (slot) {
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
            auto slot = find_slot(store->ptr());
            if (slot) {
                // If the slot has been found and is safe, try to find a value for it
                auto slot_value = get_value(slot, mapping);
                auto stored_value = insert_to_slot(store->ptr(), slot_value, store->val(), store->debug());
                mapping[slot] = stored_value;
            }
            return store->out_mem();
        } else if (auto enter = mem_use->isa<Enter>()) {
            // Loop through all slots allocated through the returned frame
            auto frame = enter->out_frame();
            for (auto use : frame->uses()) {
                // All the slots allocated at that point contain bottom
                auto slot = use->isa<Slot>();
                if (slot && is_safe_slot(slot))
                    mapping[slot] = world_.bottom(slot->type()->as<PtrType>()->pointee());
            }
            return enter->out_mem();
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
        return mapping[alloc] = world_.top(alloc->type()->as<PtrType>()->pointee(), alloc->debug());
    }

    const Def* extract_from_slot(const Def* ptr, const Def* slot_value, Debug dbg) {
        while (auto bitcast = ptr->isa<Bitcast>())
            ptr = bitcast->from();
        if (auto lea = ptr->isa<LEA>())
            return world_.extract(extract_from_slot(lea->ptr(), slot_value, dbg), lea->index(), dbg);
        return slot_value;
    }

    const Def* insert_to_slot(const Def* ptr, const Def* slot_value, const Def* insert_value, Debug dbg) {
        std::vector<const Def*> indices;
        while (true) {
            if (auto bitcast = ptr->isa<Bitcast>()) {
                ptr = bitcast->from();
            } else if (auto lea = ptr->isa<LEA>()) {
                indices.push_back(lea->index());
                ptr = lea->ptr();
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
        } else if (auto primop = def->isa<PrimOp>()) {
            for (auto op : primop->ops()) {
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
        if (auto lea = ptr->isa<LEA>()) return find_slot(lea->ptr());
        if (auto bitcast = ptr->isa<Bitcast>()) return find_slot(bitcast->from());
        return nullptr;
    }

    static void replace_ptr_uses(const Def* ptr) {
        for (auto& use : ptr->uses()) {
            if (auto store = use->isa<Store>()) {
                store->replace(store->mem());
            } else if (use->isa<Load>()) {
                assert(false);
            } else if (use->isa<LEA>() || use->isa<Bitcast>()) {
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
            } else if (use->isa<LEA>()) {
                if (!are_ptr_uses_safe(use.def(), allow_load)) return false;
            } else if (auto bitcast = use->isa<Bitcast>()) {
                // Support cast between pointers to definite and indefinite arrays
                auto ptr_to   = bitcast->type()->isa<PtrType>();
                auto ptr_from = bitcast->from()->type()->isa<PtrType>();
                if (!ptr_to || !ptr_from)
                    return false;
                auto array_to   = ptr_to->pointee()->isa<IndefiniteArrayType>();
                auto array_from = ptr_from->pointee()->isa<DefiniteArrayType>();
                if (!array_to || !array_from)
                    return false;
                if (array_to->elem_type() != array_from->elem_type())
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
