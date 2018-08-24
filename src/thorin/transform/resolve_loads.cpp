#include "thorin/analyses/scope.h"
#include "thorin/analyses/schedule.h"
#include "thorin/world.h"

namespace thorin {

static void replace_ptr_uses(const Def* ptr) {
    for (auto& use : ptr->uses()) {
        if (auto store = use->isa<Store>()) {
            store->replace(store->mem());
        } else if (use->isa<Load>()) {
            assert(false);
        } else if (use->isa<LEA>()) {
            replace_ptr_uses(use.def());
        } else if (use->isa<Bitcast>()) {
            replace_ptr_uses(use.def());
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

static bool is_safe_slot(DefMap<bool>& safe_slots, const Def* slot) {
    assert(slot->isa<Slot>());
    if (safe_slots.contains(slot)) return safe_slots[slot];
    return safe_slots[slot] = are_ptr_uses_safe(slot);
}

static const Def* remove_bitcasts(const Def* ptr) {
    while (true) {
        if (auto bitcast = ptr->isa<Bitcast>()) {
            ptr = bitcast->from();
            continue;
        } else if (auto lea = ptr->isa<LEA>()) {
            ptr = lea->rebuild({ remove_bitcasts(lea->ptr()), lea->index() });
        }
        break;
    }
    return ptr;
}

static std::pair<const Def*, size_t> find_slot(DefMap<bool>& safe_slots, const Def* ptr, size_t depth = 0) {
    if (ptr->isa<Slot>() && is_safe_slot(safe_slots, ptr)) return std::make_pair(ptr, depth);
    if (auto lea = ptr->isa<LEA>()) return find_slot(safe_slots, lea->ptr(), depth + 1);
    if (auto bitcast = ptr->isa<Bitcast>()) return find_slot(safe_slots, bitcast->from(), depth);
    return std::make_pair(nullptr, 0);
}

static inline const Def* find_base_ptr(const Def* ptr, size_t depth) {
    while (depth > 0) {
        auto lea = remove_bitcasts(ptr)->as<LEA>();
        ptr = lea->ptr();
        depth--;
    }
    return ptr;
}

static inline bool is_pointer_prefix(const Def* ptr1, size_t depth1, const Def* ptr2, size_t depth2) {
    bool dir = depth2 >= depth1;
    return remove_bitcasts(find_base_ptr(dir ? ptr2 : ptr1, dir ? depth2 - depth1 : depth1 - depth2)) == remove_bitcasts(dir ? ptr1 : ptr2);
}


static const Def* extract_from_leas(const Def* value, const Def* ptr, size_t depth) {
    if (depth == 0)
        return value;
    ptr = remove_bitcasts(ptr);
    auto& world = value->world();
    return world.extract(extract_from_leas(value, ptr->as<LEA>()->ptr(), depth - 1), ptr->as<LEA>()->index());
}

static const Def* insert_from_leas(const Def* value, const Def* elem, const Def* ptr, size_t depth) {
    if (depth == 0)
        return elem;
    ptr = remove_bitcasts(ptr);
    auto& world = value->world();
    const Def* extracts = extract_from_leas(value, ptr->as<LEA>()->ptr(), depth - 1);
    return world.insert(extracts, ptr->as<LEA>()->index(), insert_from_leas(value, elem, ptr->as<LEA>()->ptr(), depth - 1));
}

static const Def* try_resolve_load(DefMap<bool>& safe_slots, const Def* def, const Load* target_load, const Def* slot, size_t depth) {
    auto& world = def->world();
    while (true) {
        auto mem_op = def->isa<MemOp>();
        if (!mem_op)
            return nullptr;

        const Def* parent = nullptr;
        auto load  = Load::is_out_mem(mem_op->mem());
        auto store = mem_op->mem()->isa<Store>();
        if (load || store) {
            auto parent_ptr   = load ? load->ptr() : store->ptr();
            auto parent_slot  = find_slot(safe_slots, parent_ptr);
            auto parent_depth = parent_slot.second;
            if (parent_slot.first == slot && is_pointer_prefix(target_load->ptr(), depth, parent_ptr, parent_slot.second)) {
                if (depth >= parent_depth) {
                    const Def* value = load ? world.extract(load, world.literal_qs32(1, Debug())) : store->val();
                    return extract_from_leas(value, target_load->ptr(), depth - parent_depth);
                } else if (store) {
                    const Def* value = try_resolve_load(safe_slots, store, target_load, slot, depth);
                    if (!value)
                        return nullptr;
                    return insert_from_leas(value, store->ptr(), store->val(), parent_depth - depth);
                }
            }
            parent = load ? load->as<Def>() : store;
        } else if (auto enter = Enter::is_out_mem(mem_op->mem())) {
            // TODO: Solve undefined allocs (need to inspect the frame)
            parent = enter;
        } else {
            return nullptr;
        }

        def = parent;
    }
}

static bool resolve_loads(DefMap<bool>& safe_slots, const Scope& scope) {
    auto& world = scope.world();
    bool todo = false;
    for (const auto& block : schedule(scope, Schedule::Late)) {
        for (auto primop : block) {
            // Compute whether this slot is safe to transform
            auto slot = primop->isa<Slot>();
            if (slot) {
                is_safe_slot(safe_slots, slot);
                continue;
            }

            auto load = primop->isa<Load>();
            if (!load)
                continue;
            // Reolve loads that are from a safe slot
            auto slot_depth = find_slot(safe_slots, load->ptr());
            if (!slot_depth.first)
                continue;
            if (auto value = try_resolve_load(safe_slots, load, load, slot_depth.first, slot_depth.second)) {
                load->replace(world.tuple({ load->mem(), value }, load->debug()));
                todo = true;
            }
        }
    }
    return todo;
}

bool resolve_loads(World& world) {
    DefMap<bool> safe_slots;
    bool todo = false;
    Scope::for_each(world, [&] (const Scope& scope) {
        todo |= resolve_loads(safe_slots, scope);
    });
    // Remove slots that only have stores
    for (auto& pair : safe_slots) {
        if (!pair.second) continue;        
        if (are_ptr_uses_safe(pair.first, false)) {
            replace_ptr_uses(pair.first);
            todo = true;
        }
    }
    return todo;
}

} // namespace thorin
