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
        for (auto lam : world_.copy_lams()) {
            for (auto param : lam->params()) {
                if (param->type()->isa<MemType>()) {
                    Def2Def mapping;
                    resolve_loads(param, mapping);
                }
            }
        }
        return todo_;
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
                    load->replace_uses(world_.tuple({ load->mem(), load_value }));
                }
            }
            return load->out_mem();
        } else if (auto store = mem_use->isa<Store>()) {
            // Try to find the slot corresponding to this store
            auto slot = find_slot(store->ptr());
            if (slot) {
                if (only_stores(slot)) {
                    store->replace_uses(store->mem());
                } else {
                    // If the slot has been found and is safe, try to find a value for it
                    auto slot_value = get_value(slot, mapping);
                    auto stored_value = insert_to_slot(store->ptr(), slot_value, store->val(), store->debug());
                    mapping[slot] = stored_value;
                }
            }
            return store->out_mem();
        } else if (auto enter = mem_use->isa<Enter>()) {
            // Loop through all slots allocated through the returned frame
            auto frame = enter->out_frame();
            for (auto use : frame->uses()) {
                // All the slots allocated at that point contain bottom
                assert(use->isa<Slot>());
                mapping[use.def()] = world_.bottom(use->type()->as<PtrType>()->pointee());
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

    static bool is_safe_bitcast(const Bitcast* bitcast) {
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
        return true;
    }

#define CACHED(name, ...) \
private: \
    DefMap<bool> name##_; \
public: \
    bool name##_uncached(const Def* def) { \
        __VA_ARGS__ \
    } \
    bool name(const Def* def) { \
        auto it = name##_.find(def); \
        if (it != name##_.end()) \
            return it->second; \
        auto val = name##_uncached(def); \
        name##_[def] = val; \
        return val; \
    }

    CACHED(contains_top, {
        if (def->isa<Top>()) {
            return true;
        } else if (def->isa_structural()) {
            for (auto op : def->ops()) {
                if (contains_top(op))
                    return true;
            }
            return false;
        } else {
            return false;
        }
    })

    CACHED(safe_ptr, {
        // All uses have to be transitively load/store/lea/bitcast
        for (auto& use : def->uses()) {
            if (use->isa<Store>()) {
                if (use.index() != 1) return false;
            } else if (use->isa<LEA>()) {
                if (!safe_ptr(use.def())) return false;
            } else if (auto bitcast = use->isa<Bitcast>()) {
                if (!is_safe_bitcast(bitcast) || !safe_ptr(use.def())) return false;
            } else if (!use->isa<Load>()) {
                return false;
            }
        }
        return true;
    })

    CACHED(partially_safe_ptr, {
        // All uses have to be load/store/lea/bitcast but do not recurse past one lea level
        for (auto& use : def->uses()) {
            if (use->isa<Store>()) {
                if (use.index() != 1) return false;
            } else if (auto bitcast = use->isa<Bitcast>()) {
                if (!is_safe_bitcast(bitcast) || !partially_safe_ptr(use.def())) return false;
            } else if (!use->isa<Load>() && !use->isa<LEA>()) {
                return false;
            }
        }
        return true;
    })

    CACHED(only_stores, {
        // All uses have to be transitively  store/lea/bitcast
        for (auto& use : def->uses()) {
            if (use->isa<Store>()) {
                if (use.index() != 1) return false;
            } else if (auto bitcast = use->isa<Bitcast>()) {
                if (!is_safe_bitcast(bitcast) || !only_stores(use.def())) return false;
            } else if (use->isa<LEA>()) {
                if (!only_stores(use.def())) return false;
            } else {
                return false;
            }
        }
        return true;
    })

#undef CACHED

    const Def* find_slot(const Def* ptr, bool must_be_safe = true) {
        const Def* first = ptr;
        while (true) {
            while (auto bitcast = ptr->isa<Bitcast>())
                ptr = bitcast->from();
            if (ptr->isa<Global>() && !ptr->as<Global>()->is_mutable())
                return ptr;
            // If first == ptr, we are looking at the pointed value.
            // In that case, we need to make sure the pointer does not escape.
            // Otherwise, we only need to make sure that the enclosing object is not escaping,
            // but we do not have to care about its *other* children.
            auto safe = first == ptr ? safe_ptr(ptr) : partially_safe_ptr(ptr);
            if (must_be_safe && !safe)
                break;
            if (ptr->isa<Slot>())
                return ptr;
            if (auto lea = ptr->isa<LEA>()) {
                ptr = lea->ptr();
                continue;
            }
            break;
        }
        return nullptr;
    }

private:
    bool todo_;
    World& world_;
};

bool resolve_loads(World& world) {
    return ResolveLoads(world).resolve_loads();
}

} // namespace thorin
