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

    Scope::for_each(world(), [&](const Scope& scope) { emit_scope(scope); });

    // build root node with the top level stuff that got emitted
    auto root = shady::root(arena, (shady::Root) {
        .declarations = shady::nodes(arena, top_level.size(), const_cast<const shady::Node**>(top_level.data())),
    });

    char* bufptr;
    size_t size;
    shady::print_node_into_str(root, &bufptr, &size);
    out.write(bufptr, static_cast<std::streamsize>(size));
    free(bufptr);

    shady::destroy_arena(arena);
    arena = nullptr;
    top_level.clear();
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

static inline int find_return_parameter(const FnType* type) {
    for (size_t i = 0; i < type->num_ops(); i++) {
        auto t = type->op(i);
        if (t->order() % 2 == 1)
            return static_cast<int>(i);
    }
    return -1;
}

const shady::Type* CodeGen::convert(const Type* type) {
    if (auto res = types_.lookup(type)) return *res;
    const shady::Type* t;
    if (type == world().mem_type()) {
        t = nullptr;
        goto skip_check;
    } else if (auto prim = type->isa<PrimType>()) {
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
    } else if (auto fn_type = type->isa<FnType>()) {
        shady::FnType payload = {};
        payload.is_basic_block = fn_type->is_basicblock();
        NodeVec dom, codom;

        int return_param_i = find_return_parameter(fn_type);
        for (size_t i = 0; i < fn_type->num_ops(); i++) {
            // Skip the return param
            if (return_param_i != -1 && i == static_cast<size_t>(return_param_i)) continue;
            auto converted = convert(fn_type->op(i));
            if (!converted)
                continue; // Eliminate mem params
            shady::QualifiedType qtype;
            qtype.type = converted;
            qtype.is_uniform = false;
            converted = shady::qualified_type(arena, qtype);
            dom.push_back(converted);
        }

        if (return_param_i != -1) {
            auto ret_fn_type = fn_type->op(return_param_i);
            for (size_t i = 0; i < ret_fn_type->num_ops(); i++) {
                auto converted = convert(ret_fn_type->op(i));
                if (!converted)
                    continue; // Eliminate mem params
                codom.push_back(converted);
            }
        }

        if (fn_type->is_basicblock())
            assert(codom.empty());
        payload.param_types = vec2nodes(dom);
        payload.return_types = vec2nodes(codom);
        t = shady::fn_type(arena, payload);
    } else {
        assert(false);
    }

    assert(t);
    skip_check:
    types_[type] = t;
    return t;
}

shady::Node* CodeGen::def_to_decl(Def* def) {
    NodeVec annotations;
    if (auto cont = def->isa_nom<Continuation>()) {
        NodeVec params;
        NodeVec returns;

        int ret_param_i = find_return_parameter(cont->type());

        for (size_t i = 0; i < cont->num_params(); i++) {
            if (ret_param_i != -1 && i == static_cast<size_t>(ret_param_i))
                continue; // Skip the return parameter
            auto type = convert(cont->param(i)->type());
            if (!type) continue; // Eliminate mem tokens
            shady::QualifiedType qtype;
            qtype.type = type;
            qtype.is_uniform = false;
            type = shady::qualified_type(arena, qtype);
            auto param = shady::var(arena, type, cont->param(i)->name().c_str());
            defs_[cont->param(i)] = param; // Register the param as emitted already
            params.push_back(param);
        }

        if (!cont->is_basicblock() && ret_param_i >= 0) {
            auto ret_fn_type = cont->type()->op(ret_param_i);

            for (auto t : ret_fn_type->ops()) {
                auto ret_type = convert(t);
                if (!ret_type)
                    continue; // Eliminate mem types
                returns.push_back(ret_type);
            }
        }

        return shady::fn(arena, vec2nodes(annotations), def->unique_name().c_str(), cont->is_basicblock(), vec2nodes(params), vec2nodes(returns));
    } else if (auto global = def->isa<Global>()) {
        if (global->is_mutable()) {
            return shady::global_var(arena, vec2nodes(annotations), convert(global->alloced_type()), global->unique_name().c_str(), convert_address_space(AddrSpace::Private));
        } else {
            // Tentatively make those things constants...
            auto constant = shady::constant(arena, vec2nodes(annotations), global->unique_name().c_str());;
            constant->payload.constant.type_hint = convert(global->alloced_type());
            return constant;
        }
    } else {
        assert(false && "This doesn't map to a decl !");
    }
}

shady::Node* CodeGen::get_decl(Def* def) {
    for (auto& e : top_level) {
        if (shady::get_decl_name(e) == def->unique_name())
            return e;
    }

    auto decl = def_to_decl(def);
    top_level.push_back(decl);
    return decl;
}

shady::Node* CodeGen::prepare(const Scope& scope) {
    cont2bb_[scope.entry()].head = curr_fn = get_decl(scope.entry());
}

void CodeGen::prepare(Continuation* cont, shady::Node*) {
    BB& bb = cont2bb_[cont];
    if (cont->is_basicblock())
        bb.head = def_to_decl(cont);
    else
        assert(bb.head);

    // Register params
    // for (size_t i = 0; i < cont->num_params(); i++)
    //     defs_[cont->param(i)] = bb.head->payload.fn.params.nodes[i];

    bb.builder = shady::begin_block(arena);
}

void CodeGen::emit_epilogue(Continuation* cont) {
    BB& bb = cont2bb_[cont];
    assert(cont->has_body());
    auto body = cont->body();
    NodeVec args;
    for (auto& arg : body->args()) {
        if (convert(arg->type()) == nullptr) {
            args.push_back(nullptr);
        } else if (auto callee = arg->isa_nom<Continuation>(); callee && callee->is_basicblock()) {
            // Emitting basic blocks as values isn't legal - but for convenience we'll put them in our list.
            BB& callee_bb = cont2bb_[callee];
            assert(callee_bb.head && callee_bb.head->payload.fn.is_basic_block);
            args.push_back(callee_bb.head);
        } else {
            args.push_back(emit(arg));
        }
    }

    if (body->callee() == entry_->ret_param()) {
        shady::Return payload = {};
        payload.fn = curr_fn;
        args.erase(std::remove_if(args.begin(), args.end(), [&](const auto& item){ return item == nullptr || !shady::is_value(item); }), args.end());
        payload.values = vec2nodes(args);
        bb.terminator = shady::fn_ret(arena, payload);
    } else if (body->callee() == world().branch()) {
        shady::Branch payload = {};
        payload.branch_mode = shady::Branch::BrIfElse;
        payload.args = shady::nodes(arena, 0, nullptr);
        payload.branch_condition = args[0];
        payload.true_target      = args[1];
        payload.false_target     = args[2];
        bb.terminator = shady::branch(arena, payload);
    } else if (auto match = body->callee()->as_nom<Continuation>(); match && match->intrinsic() == Intrinsic::Match) {
        assert(false);
    } else if (auto destination = body->callee()->isa_nom<Continuation>(); destination && destination->is_basicblock()) {
        shady::Branch payload = {};
        args.erase(std::remove_if(args.begin(), args.end(), [&](const auto& item){ return item == nullptr || !shady::is_value(item); }), args.end());
        payload.args = vec2nodes(args);
        payload.branch_mode = shady::Branch_::BrJump;
        bb.terminator = shady::branch(arena, payload);
    } else if (auto intrinsic = body->callee()->isa_nom<Continuation>(); intrinsic && intrinsic->is_intrinsic()) {
        assert(false);
    } else if (auto callee = body->callee()->isa_nom<Continuation>()) {
        shady::Callc payload = {};
        int ret_param = find_return_parameter(callee->type());
        assert(ret_param >= 0);
        payload.ret_cont = args[ret_param];
        args.erase(args.begin() + ret_param);
        args.erase(std::remove_if(args.begin(), args.end(), [&](const auto& item){ return item == nullptr || !shady::is_value(item); }), args.end());
        payload.args = vec2nodes(args);
        payload.is_return_indirect = false;
        payload.callee = emit(callee);
        bb.terminator = shady::callc(arena, payload);
    } else {
        assert(false);
    }
}

void CodeGen::finalize(Continuation* cont) {
    BB& bb = cont2bb_[cont];
    assert(bb.head && bb.builder && bb.terminator);
    bb.block = shady::finish_block(bb.builder, bb.terminator);
    bb.head->payload.fn.block = bb.block;
}

void CodeGen::finalize(const Scope& scope) {
    BB& bb = cont2bb_[scope.entry()];
    assert(bb.head->payload.fn.block != nullptr);
    curr_fn = nullptr;
}

const shady::Node* CodeGen::emit_fun_decl(Continuation* cont) {
    assert(!cont->is_basicblock());
    shady::FnAddr payload;
    payload.fn = get_decl(cont);
    return shady::fn_addr(arena, payload);
}

const shady::Node* CodeGen::emit_bb(BB& bb, const Def* def) {
    const shady::Node* v = nullptr;
    if (auto prim_lit = def->isa<PrimLit>()) {
        const auto& box = prim_lit->value();
        switch (prim_lit->primtype_tag()) {
            case PrimType_bool:                     v = box.get_bool() ? shady::true_lit(arena) : shady::false_lit(arena); break;
            case PrimType_ps8:  case PrimType_qs8:  v =  shady::int8_literal (arena,  box.get_s8()); break;
            case PrimType_pu8:  case PrimType_qu8:  v = shady::uint8_literal (arena,  box.get_u8()); break;
            case PrimType_ps16: case PrimType_qs16: v =  shady::int16_literal(arena, box.get_s16()); break;
            case PrimType_pu16: case PrimType_qu16: v = shady::uint16_literal(arena, box.get_u16()); break;
            case PrimType_ps32: case PrimType_qs32: v =  shady::int32_literal(arena, box.get_s32()); break;
            case PrimType_pu32: case PrimType_qu32: v = shady::uint32_literal(arena, box.get_u32()); break;
            case PrimType_ps64: case PrimType_qs64: v =  shady::int64_literal(arena, box.get_s64()); break;
            case PrimType_pu64: case PrimType_qu64: v = shady::uint64_literal(arena, box.get_u64()); break;
            case PrimType_pf16: case PrimType_qf16: assert(false && "TODO");
            case PrimType_pf32: case PrimType_qf32: v = shady::float_type(arena); break;
            case PrimType_pf64: case PrimType_qf64: assert(false && "TODO");
            default: THORIN_UNREACHABLE;
        }
    } else if (auto arr = def->isa<DefiniteArray>()) {
        NodeVec contents;
        for (auto& e : arr->ops()) {
            assert(emit(e));
            contents.push_back(emit(e));
        }
        shady::ArrayLiteral payload;
        payload.element_type = convert(arr->elem_type());
        payload.contents = vec2nodes(contents);
        v = shady::arr_lit(arena, payload);
    }
    assert(v && shady::is_value(v));
    defs_[def] = v;
    return v;
}

}
