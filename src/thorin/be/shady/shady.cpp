#include "shady.h"

#include "thorin/analyses/scope.h"
#include "thorin/transform/structurize.h"

namespace thorin::shady_be {

CodeGen::CodeGen(thorin::World& world, Cont2Config& kernel_config, bool debug)
        : thorin::CodeGen(world, debug), kernel_config_(kernel_config)
{}

void CodeGen::emit_stream(std::ostream& out) {
    assert(top_level.empty());

    structure_loops(world());
    structure_flow(world());

    auto config = shady::ArenaConfig {
        .check_types = true,
    };
    arena = shady::new_arena(config);

    Scope::for_each(world(), [&](const Scope& scope) { emit(scope); });

    // build root node with the top level stuff that got emitted
    auto decls = std::vector<shady::Node*>(top_level.size(), nullptr);
    for (size_t i = 0; i < top_level.size(); i++) {
        decls[i] = top_level[i].first;
    }
    auto root = shady::root(arena, (shady::Root) {
        .declarations = shady::nodes(arena, top_level.size(), const_cast<const shady::Node**>(decls.data())),
    });

    shady::print_node(root);

    out << "todo";

    shady::destroy_arena(arena);
    arena = nullptr;
    top_level.clear();
}

void CodeGen::emit(const thorin::Scope& scope) {
    entry_ = scope.entry();
    assert(entry_->is_returning());

    assert(false && "TODO");
}

shady::AddressSpace CodeGen::convert_address_space(AddrSpace as) {
    switch(as) {
        case AddrSpace::Generic: assert(false); break;
        case AddrSpace::Global: return shady::AsGlobalPhysical;
        case AddrSpace::Texture: assert(false); break;
        case AddrSpace::Shared: return shady::AsSharedPhysical;
        case AddrSpace::Constant: assert(false); break;
        case AddrSpace::Private: return shady::AsPrivatePhysical;
    }
    assert(false);
}

const shady::Type* CodeGen::convert(const Type* type) {
    if (auto res = types_.lookup(type)) return *res;
    const shady::Type* t;
    if (auto prim = type->isa<PrimType>()) {
        switch (prim->primtype_tag()) {
            case PrimType_bool:                     t = shady::bool_type(arena); break;
            case PrimType_ps8:  case PrimType_qs8:
            case PrimType_pu8:  case PrimType_qu8:  t = shady::int8_type(arena); break;
            case PrimType_ps16: case PrimType_qs16:
            case PrimType_pu16: case PrimType_qu16: t = shady::int16_type(arena); break;
            case PrimType_ps32: case PrimType_qs32:
            case PrimType_pu32: case PrimType_qu32: t = shady::int32_type(arena); break;
            case PrimType_ps64: case PrimType_qs64:
            case PrimType_pu64: case PrimType_qu64: t = shady::int64_type(arena); break;
            case PrimType_pf16: case PrimType_qf16: assert(false && "TODO");
            case PrimType_pf32: case PrimType_qf32: t = shady::float_type(arena); break;
            case PrimType_pf64: case PrimType_qf64: assert(false && "TODO");
            default: THORIN_UNREACHABLE;
        }
    } else if (auto ptr = type->isa<PtrType>()) {
        t = shady::ptr_type(arena, (shady::PtrType) {
            convert_address_space(ptr->addr_space()),
            convert(ptr->pointee())
        });
    } else if (auto arr = type->isa<ArrayType>()) {
        shady::ArrType payload = {};
        payload.element_type = convert(arr->elem_type());
        payload.size = nullptr;
        if (auto definite = arr->isa<DefiniteArrayType>()) {
            payload.size = shady::int32_literal(arena, static_cast<uint32_t>(definite->dim()));
        }
        t = shady::arr_type(arena, payload);
    } else if (auto strct = type->isa<StructType>()) {
        auto members = std::vector<const shady::Type*>(strct->num_ops());
        for (size_t i = 0; i < strct->num_ops(); i++) {
            members[i] = convert(strct->op(i));
        }
        shady::RecordType payload = {};
        payload.members = shady::nodes(arena, strct->num_ops(), members.data());
        payload.names = shady::strings(arena, 0, nullptr);
        payload.special = shady::RecordType::NotSpecial;
        t = shady::record_type(arena, payload);
    } else if (auto variant = type->isa<VariantType>()) {
        assert(false && "TODO");
    } else {
        assert(false);
    }

    assert(t);
    types_[type] = t;
    return t;
}

}
