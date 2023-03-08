#include "shady.h"
#undef empty

#include "thorin/analyses/scope.h"

namespace thorin::shady_be {

CodeGen::CodeGen(thorin::Thorin& thorin, Cont2Config& kernel_config, bool debug)
        : thorin::CodeGen(thorin, debug), kernel_config_(kernel_config)
{}

void CodeGen::emit_stream(std::ostream& out) {
    assert(!module);

    shady::ArenaConfig config = shady::default_arena_config();
    config.name_bound = true;
    config.check_types = true;
    arena = shady::new_ir_arena(config);
    module = shady::new_module(arena, world().name().c_str());

    ScopesForest(world()).for_each([&](const Scope& scope) { emit_scope(scope); });

    char* bufptr;
    size_t size;
    shady::print_module_into_str(module, &bufptr, &size);
    out.write(bufptr, static_cast<std::streamsize>(size));
    free(bufptr);

    shady::destroy_ir_arena(arena);
    arena = nullptr;
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
        shady::PtrType payload = {
            convert_address_space(ptr->addr_space()),
            convert(ptr->pointee())
        };
        t = shady::ptr_type(arena, payload);
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
            members[i] = convert(strct->types()[i]);
        }
        shady::RecordType payload = {};
        payload.members = shady::nodes(arena, strct->num_ops(), members.data());
        payload.names = shady::strings(arena, 0, nullptr);
        payload.special = shady::NotSpecial;
        t = shady::record_type(arena, payload);
    } else if (auto variant = type->isa<VariantType>()) {
        assert(false && "TODO");
    } else if (auto fn_type = type->isa<FnType>()) {
        shady::FnType payload = {};
        NodeVec dom, codom;

        int return_param_i = find_return_parameter(fn_type);
        for (size_t i = 0; i < fn_type->num_ops(); i++) {
            // Skip the return param
            if (return_param_i != -1 && i == static_cast<size_t>(return_param_i)) continue;
            auto converted = convert(fn_type->types()[i]);
            if (!converted)
                continue; // Eliminate mem params
            shady::QualifiedType qtype;
            qtype.type = converted;
            qtype.is_uniform = false;
            converted = shady::qualified_type(arena, qtype);
            dom.push_back(converted);
        }

        if (return_param_i != -1) {
            auto ret_fn_type = fn_type->types()[return_param_i]->as<FnType>();
            for (size_t i = 0; i < ret_fn_type->num_ops(); i++) {
                auto converted = convert(ret_fn_type->types()[i]);
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

shady::Node* CodeGen::emit_decl_head(Def* def) {
    NodeVec annotations;
    if (auto cont = def->isa_nom<Continuation>()) {
        assert(!cont->is_basicblock());
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

        auto ret_fn_type = cont->type()->types()[ret_param_i]->as<FnType>();

        for (auto t : ret_fn_type->types()) {
            auto ret_type = convert(t);
            if (!ret_type)
                continue; // Eliminate mem types

            shady::QualifiedType qtype;
            qtype.type = ret_type;
            qtype.is_uniform = false;
            ret_type = shady::qualified_type(arena, qtype);
            returns.push_back(ret_type);
        }

        std::string name = def->unique_name();

        auto config = kernel_config_.find(cont);
        if (config != kernel_config_.end()) {
            if (auto gpu_config = config->second->isa<GPUKernelConfig>()) {
                annotations.push_back(shady::annotation_value(arena, { .name = "EntryPoint", .value = shady::string_lit(arena, { .string = "compute" })}));
                std::vector<const shady::Node*> block_size;
                block_size.emplace_back(shady::int32_literal(arena, get<0>(gpu_config->block_size())));
                block_size.emplace_back(shady::int32_literal(arena, get<1>(gpu_config->block_size())));
                block_size.emplace_back(shady::int32_literal(arena, get<2>(gpu_config->block_size())));
                annotations.push_back(shady::annotation_values(arena, { .name = "WorkgroupSize", .values = vec2nodes(block_size) }));
                name = "main";
            } else {
                assert(false && "Only GPU kernel configs are currently supported");
            }
        }

        return shady::function(module, vec2nodes(params), name.c_str(), vec2nodes(annotations), vec2nodes(returns));
    } else if (auto global = def->isa<Global>()) {
        if (global->is_mutable()) {
            return shady::global_var(module, vec2nodes(annotations), convert(global->alloced_type()), global->unique_name().c_str(), convert_address_space(AddrSpace::Private));
        } else {
            // Tentatively make those things constants...
            return shady::constant(module, vec2nodes(annotations), convert(global->alloced_type()), global->unique_name().c_str());
        }
    } else {
        assert(false && "This doesn't map to a decl !");
    }
}

shady::Node* CodeGen::get_decl(Def* def) {
    shady::Nodes already_done = shady::get_module_declarations(module);
    for (size_t i = 0; i < already_done.count; i++) {
        auto& e = already_done.nodes[i];
        if (shady::get_decl_name(e) == def->unique_name())
            return (shady::Node*) e;
    }

    return emit_decl_head(def);
}

shady::Node* CodeGen::prepare(const Scope& scope) {
    return cont2bb_[scope.entry()].head = curr_fn = get_decl(scope.entry());
}

void CodeGen::prepare(Continuation* cont, shady::Node*) {
    BB& bb = cont2bb_[cont];
    if (cont->is_basicblock()) {
        NodeVec params;

        for (size_t i = 0; i < cont->num_params(); i++) {
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

        bb.head = shady::basic_block(arena, curr_fn, vec2nodes(params), cont->unique_name().c_str());
    } else
        assert(bb.head);

    bb.builder = shady::begin_body(module);
}

static std::optional<shady::Op> is_shady_prim_op(const Continuation* cont) {
    for (int i = 0; i < shady::PRIMOPS_COUNT; i++) {
        if (cont->name() == shady::primop_names[i])
            return std::make_optional((shady::Op) i);
    }
    return std::nullopt;
}

static std::vector<const shady::Node*> emit_instruction(shady::BodyBuilder* builder, const shady::Node* instruction);

void CodeGen::emit_epilogue(Continuation* cont) {
    BB& bb = cont2bb_[cont];
    assert(cont->has_body());
    auto body = cont->body();
    NodeVec args;
    for (auto& arg : body->args()) {
        if (convert(arg->type()) == nullptr) {
            args.push_back(nullptr);
        } else if (auto target = arg->isa_nom<Continuation>(); target && target->is_basicblock()) {
            // Emitting basic blocks as values isn't legal - but for convenience we'll put them in our list.
            assert(cont2bb_.contains(target));
            BB& callee_bb = cont2bb_[target];
            assert(callee_bb.head && shady::is_basic_block(callee_bb.head));
            args.push_back(callee_bb.head);
        } else {
            args.push_back(emit(arg));
        }
    }

    if (body->callee() == entry_->ret_param()) {
        shady::Return payload = {};
        payload.fn = curr_fn;
        args.erase(std::remove_if(args.begin(), args.end(), [&](const auto& item){ return item == nullptr || !shady::is_value(item); }), args.end());
        payload.args = vec2nodes(args);
        bb.terminator = shady::fn_ret(arena, payload);
    } else if (body->callee() == world().branch()) {
        shady::Branch payload = {};
        payload.args = shady::nodes(arena, 0, nullptr);
        payload.branch_condition = args[0];
        payload.true_target      = args[1];
        payload.false_target     = args[2];
        bb.terminator = shady::branch(arena, payload);
    } else if (auto match = body->callee()->as_nom<Continuation>(); match && match->intrinsic() == Intrinsic::Match) {
        assert(false);
    } else if (auto destination = body->callee()->isa_nom<Continuation>(); destination && destination->is_basicblock()) {
        shady::Jump payload = {};
        args.erase(std::remove_if(args.begin(), args.end(), [&](const auto& item){ return item == nullptr || !shady::is_value(item); }), args.end());
        payload.args = vec2nodes(args);
        payload.target = args[0];
        bb.terminator = shady::jump(arena, payload);
    } else if (auto intrinsic = body->callee()->isa_nom<Continuation>(); intrinsic && intrinsic->is_intrinsic()) {
        assert(false);
    } else {
        int ret_param = find_return_parameter(body->callee()->type()->as<FnType>());
        // TODO handle tail calls ?
        assert(ret_param >= 0);

        args.erase(args.begin() + ret_param);
        args.erase(std::remove_if(args.begin(), args.end(), [&](const auto& item){ return item == nullptr || !shady::is_value(item); }), args.end());

        if (auto callee = body->callee()->isa_nom<Continuation>()) {
            // shady primop called as imported continuations look like continuation calls to thorin, but not to shady
            // we just need to carefully emit the primop as an instruction, then jump to the target BB, passing the stuff as we do
            if (auto op = is_shady_prim_op(callee); op.has_value()) {
                shady::Jump jump;
                jump.target = args[ret_param];
                jump.args = shady::bind_instruction(bb.builder, shady::prim_op(arena, (shady::PrimOp) {
                    .op = op.value(),
                    .type_arguments = shady::nodes(arena, 0, nullptr),
                    .operands = vec2nodes(args),
                }));
                bb.terminator = shady::jump(arena, jump);
                return;
            }
        }

        shady::BodyBuilder* builder = shady::begin_body(module);

        shady::IndirectCall icall_payload;
        icall_payload.args = vec2nodes(args);
        icall_payload.callee = emit(body->callee());
        shady::Nodes results = shady::bind_instruction(builder, shady::indirect_call(arena, icall_payload));

        assert(args[ret_param]->tag == shady::BasicBlock_TAG);
        shady::Jump jump_payload;
        jump_payload.target = args[ret_param];
        jump_payload.args = results;
        bb.terminator = shady::finish_body(builder, shady::jump(arena, jump_payload));
    }
}

void CodeGen::finalize(Continuation* cont) {
    BB& bb = cont2bb_[cont];
    assert(bb.head && bb.builder && bb.terminator);
    if (shady::is_basic_block(bb.head))
        bb.head->payload.basic_block.body = shady::finish_body(bb.builder, bb.terminator);
    else if (shady::is_function(bb.head))
        bb.head->payload.fun.body = shady::finish_body(bb.builder, bb.terminator);
    else
        assert(false);
}

void CodeGen::finalize(const Scope& scope) {
    BB& bb = cont2bb_[scope.entry()];
    assert(bb.head->payload.fun.body != nullptr);
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

    auto mk_primop = [&](shady::Op op, std::vector<const Def*> args, std::vector<const Type*> types = {}) -> const shady::Node* {
        shady::PrimOp payload = {};
        payload.op = op;
        std::vector<const shady::Node*> operands;
        for (auto arg : args)
            operands.push_back(emit(arg));
        std::vector<const shady::Node*> type_arguments;
        for (auto type_arg : types)
            type_arguments.push_back(convert(type_arg));
        payload.operands = vec2nodes(operands);
        payload.type_arguments = vec2nodes(type_arguments);
        return shady::first(shady::bind_instruction(bb.builder, shady::prim_op(arena, payload)));
    };

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
        shady::ArrType payload;
        const shady::Type* arr_type = shady::arr_type(arena, payload);
        payload.element_type = convert(arr->elem_type());
        payload.size = shady::int32_literal(arena, contents.size());
        v = shady::composite(arena, arr_type, vec2nodes(contents));
    } else if (auto cmp = def->isa<Cmp>()) {
        switch (cmp->cmp_tag()) {
            case Cmp_eq: v = mk_primop(shady::Op::eq_op,  { cmp->lhs(), cmp->rhs() }); break;
            case Cmp_ne: v = mk_primop(shady::Op::neq_op, { cmp->lhs(), cmp->rhs() }); break;
            case Cmp_gt: v = mk_primop(shady::Op::gt_op,  { cmp->lhs(), cmp->rhs() }); break;
            case Cmp_ge: v = mk_primop(shady::Op::gte_op, { cmp->lhs(), cmp->rhs() }); break;
            case Cmp_lt: v = mk_primop(shady::Op::lt_op,  { cmp->lhs(), cmp->rhs() }); break;
            case Cmp_le: v = mk_primop(shady::Op::lte_op, { cmp->lhs(), cmp->rhs() }); break;
        }
    } else if (auto arith = def->isa<ArithOp>()) {
        switch (arith->arithop_tag()) {
            case ArithOp_add: v = mk_primop(shady::Op::add_op,    { arith->lhs(), arith->rhs() }); break;
            case ArithOp_sub: v = mk_primop(shady::Op::sub_op,    { arith->lhs(), arith->rhs() }); break;
            case ArithOp_mul: v = mk_primop(shady::Op::mul_op,    { arith->lhs(), arith->rhs() }); break;
            case ArithOp_div: v = mk_primop(shady::Op::div_op,    { arith->lhs(), arith->rhs() }); break;
            case ArithOp_rem: v = mk_primop(shady::Op::mod_op,    { arith->lhs(), arith->rhs() }); break;
            case ArithOp_and: v = mk_primop(shady::Op::and_op,    { arith->lhs(), arith->rhs() }); break;
            case ArithOp_or:  v = mk_primop(shady::Op::or_op,     { arith->lhs(), arith->rhs() }); break;
            case ArithOp_xor: v = mk_primop(shady::Op::xor_op,    { arith->lhs(), arith->rhs() }); break;
            case ArithOp_shl: v = mk_primop(shady::Op::lshift_op, { arith->lhs(), arith->rhs() }); break;
            case ArithOp_shr: v = mk_primop(shady::Op::rshift_logical_op, { arith->lhs(), arith->rhs() }); break;
        }
    } else if (auto param = def->isa<Param>()) {
        assert(param->type() == world().mem_type());
        defs_[def] = nullptr;
        return nullptr;
    }
    assert(v && shady::is_value(v));
    defs_[def] = v;
    return v;
}

const shady::Node* CodeGen::emit_constant(const Def* def) {
    THORIN_UNREACHABLE;
}

}
