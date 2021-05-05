#include "thorin/be/spirv/spirv.h"
#include "thorin/util/stream.h"

namespace thorin::spirv {

ScalarDatatype::ScalarDatatype(ConvertedType* type, int type_tag, size_t size_in_bytes, size_t alignment_in_bytes)
: Datatype(type), type_tag(type_tag), size_in_bytes(size_in_bytes), alignment(alignment_in_bytes)
{

}

/// All serialization/deserialization methods use this so into a macro it goes
#define serialization_types \
SpvId u32_tid = type->code_gen->convert(type->code_gen->world().type_pu32())->type_id; \
SpvId arr_cell_tid = bb.file_builder.declare_ptr_type(storage_class, u32_tid);

SpvId ScalarDatatype::emit_deserialization(BasicBlockBuilder& bb, spv::StorageClass storage_class, SpvId array, SpvId base_offset) {
    /// currently limited to 32-bit
    assert(size_in_bytes == 4);
    serialization_types;
    auto cell = bb.access_chain(arr_cell_tid, array, { base_offset });
    auto loaded = bb.load(u32_tid, cell);
    return bb.bitcast(type->type_id, loaded);
}

void ScalarDatatype::emit_serialization(BasicBlockBuilder& bb, spv::StorageClass storage_class, SpvId array, SpvId base_offset, SpvId data) {
    /// currently limited to 32-bit
    assert(size_in_bytes == 4);
    serialization_types;
    auto cell = bb.access_chain(arr_cell_tid, array, { base_offset });
    auto casted = bb.bitcast(u32_tid, data);
    bb.store(casted, cell);
}

SpvId PtrDatatype::emit_deserialization(BasicBlockBuilder& bb, spv::StorageClass storage_class, SpvId array, SpvId base_offset) {
    assert(type->src_type->as<PtrType>()->addr_space() == AddrSpace::Global && "Only buffer device address (global memory) pointers supported");
    serialization_types;
    SpvId u64_tid = type->code_gen->convert(type->code_gen->world().type_pu64())->type_id;

    auto cell0 = bb.access_chain(arr_cell_tid, array, { base_offset });
    auto cell1 = bb.access_chain(arr_cell_tid, array, { bb.binop(spv::OpIAdd, u32_tid, base_offset, bb.file_builder.constant(u32_tid, { (uint32_t) 1 })) });

    auto lower = bb.u_convert(u64_tid, bb.load(u32_tid, cell0));
    auto upper = bb.u_convert(u64_tid, bb.load(u32_tid, cell1));

    SpvId c32 = bb.file_builder.constant(u32_tid, { 32 });
    auto merged = bb.binop(spv::OpBitwiseOr, u64_tid, lower, bb.binop(spv::OpShiftLeftLogical, u64_tid, upper, c32));

    return bb.convert_u_ptr(type->type_id, merged);
}

void PtrDatatype::emit_serialization(BasicBlockBuilder& bb, spv::StorageClass storage_class, SpvId array, SpvId base_offset, SpvId data) {
    assert(false && "TODO");
}

DefiniteArrayDatatype::DefiniteArrayDatatype(ConvertedType* type, ConvertedType* element_type, size_t length) : Datatype(type), element_type(element_type), length(length) {
    assert(element_type->datatype.get() != nullptr);
    assert(length > 0 && "Array lengths of zero are not supported");
}

SpvId DefiniteArrayDatatype::emit_deserialization(BasicBlockBuilder& bb, spv::StorageClass storage_class, SpvId array, SpvId base_offset) {
    serialization_types;
    std::vector<SpvId> indices;
    std::vector<SpvId> elements;
    SpvId offset = base_offset;
    SpvId stride = bb.file_builder.constant(u32_tid, { (uint32_t) element_type->datatype->serialized_size() });
    for (size_t i = 0; i < length; i++) {
        SpvId element = element_type->datatype->emit_deserialization(bb, storage_class, array, offset);
        elements.push_back(element);
        offset = bb.binop(spv::OpIAdd, u32_tid, offset, stride);
    }
    return bb.composite(type->type_id, elements);
}
void DefiniteArrayDatatype::emit_serialization(BasicBlockBuilder& bb, spv::StorageClass storage_class, SpvId array, SpvId base_offset, SpvId data) {
    serialization_types;
    std::vector<SpvId> indices;
    SpvId offset = base_offset;
    SpvId stride = bb.file_builder.constant(u32_tid, { (uint32_t) element_type->datatype->serialized_size() });
    for (size_t i = 0; i < length; i++) {
        element_type->datatype->emit_serialization(bb, storage_class, array, offset, bb.extract(element_type->type_id, data, { (uint32_t) i }));
        offset = bb.binop(spv::OpIAdd, u32_tid, offset, stride);
    }
}

ProductDatatype::ProductDatatype(ConvertedType* type, const std::vector<ConvertedType*>&& elements_types) : Datatype(type), elements_types(elements_types) {
    // Unit datatype is acceptable, but serdes methods should never be invoked.
    for (auto& element_type : elements_types) {
        assert(element_type->datatype != nullptr);
        total_size += element_type->datatype->serialized_size();
    }
}

SpvId ProductDatatype::emit_deserialization(BasicBlockBuilder& bb, spv::StorageClass storage_class, SpvId array, SpvId base_offset) {
    assert(total_size > 0 && "It doesn't make sense to de-serialize Unit!");
    serialization_types;
    std::vector<SpvId> indices;
    std::vector<SpvId> elements;
    SpvId offset = base_offset;
    for (auto& element_type : elements_types) {
        SpvId element = element_type->datatype->emit_deserialization(bb, storage_class, array, offset);
        offset = bb.binop(spv::OpIAdd, u32_tid, offset, bb.file_builder.constant(u32_tid, { (uint32_t) element_type->datatype->serialized_size() }));
        elements.push_back(element);
    }
    return bb.composite(type->type_id, elements);
}
void ProductDatatype::emit_serialization(BasicBlockBuilder& bb, spv::StorageClass storage_class, SpvId array, SpvId base_offset, SpvId data) {
    assert(total_size > 0 && "It doesn't make sense to serialize Unit!");
    serialization_types;
    std::vector<SpvId> indices;
    SpvId offset = base_offset;
    int i = 0;
    for (auto& element_type : elements_types) {
        element_type->datatype->emit_serialization(bb, storage_class, array, offset, bb.extract(element_type->type_id, data, { (uint32_t) i++ }));
        offset = bb.binop(spv::OpIAdd, u32_tid, offset, bb.file_builder.constant(u32_tid, { (uint32_t) element_type->datatype->serialized_size() }));
    }
}

ConvertedType* CodeGen::convert(const Type* type) {
    // Spir-V requires each primitive type to be "unique", it doesn't allow for example two 32-bit signed integer types.
    // Therefore we must enforce that precise/quick types map to the same thing.
    switch (type->tag()) {
#define THORIN_Q_TYPE(T, M) \
    case PrimType_##T: \
        type = world().prim_type(PrimType_p##M, type->as<VectorType>()->length()); \
        break;
#include "thorin/tables/primtypetable.h"
#undef THORIN_Q_TYPE
        default: break;
    }

    if (auto iter = types_.find(type); iter != types_.end()) return iter->second.get();
    ConvertedType* converted = types_.emplace(type, std::make_unique<ConvertedType>(this) ).first->second.get();
    converted->src_type = type;

    if (auto vec = type->isa<VectorType>(); vec && vec->length() > 1) {
        auto component = vec->scalarize();
        auto conv_comp = convert(component);
        converted->type_id = builder_->declare_vector_type(conv_comp->type_id, (uint32_t)vec->length());
        return converted;
    }

    switch (type->tag()) {
        // Boolean types are typically packed intelligently when declaring in local variables, however with vanilla Vulkan 1.0 they can only be represented via 32-bit integers
        // Using extensions, we could use 16 or 8-bit ints instead
        // We can also pack them inside structures using bit-twiddling tricks, if the need arises
        // Note: this only affects storing booleans inside structures, for regular variables the actual spir-v bool type is used.
        case PrimType_bool:
            converted->type_id = builder_->declare_bool_type();
            converted->datatype = std::make_unique<ScalarDatatype>(converted, type->tag(), 1, 1);
            break;
        case PrimType_ps8:  assert(false && "TODO: look into capabilities to enable this");
        case PrimType_pu8:  assert(false && "TODO: look into capabilities to enable this");
        case PrimType_ps16:
            converted->type_id = builder_->declare_int_type(16, true);
            converted->datatype = std::make_unique<ScalarDatatype>(converted, type->tag(), 2, 2);
            break;
        case PrimType_pu16:
            converted->type_id = builder_->declare_int_type(16, false);
            converted->datatype = std::make_unique<ScalarDatatype>(converted, type->tag(), 2, 2);
            break;
        case PrimType_ps32:
            converted->type_id = builder_->declare_int_type(32, true );
            converted->datatype = std::make_unique<ScalarDatatype>(converted, type->tag(), 4, 4);
            break;
        case PrimType_pu32:
            converted->type_id = builder_->declare_int_type(32, false);
            converted->datatype = std::make_unique<ScalarDatatype>(converted, type->tag(), 4, 4);
            break;
        case PrimType_ps64:
            converted->type_id = builder_->declare_int_type(64, true);
            converted->datatype = std::make_unique<ScalarDatatype>(converted, type->tag(), 8, 8);
            break;
        case PrimType_pu64:
            converted->type_id = builder_->declare_int_type(64, false);
            converted->datatype = std::make_unique<ScalarDatatype>(converted, type->tag(), 8, 8);
            break;
        case PrimType_pf16: assert(false && "TODO: look into capabilities to enable this");
        case PrimType_pf32:
            converted->type_id = builder_->declare_float_type(32);
            converted->datatype = std::make_unique<ScalarDatatype>(converted, type->tag(), 4, 4);
            break;
        case PrimType_pf64: assert(false && "TODO: look into capabilities to enable this");
        case Node_PtrType: {
            auto ptr = type->as<PtrType>();
            spv::StorageClass storage_class;
            switch (ptr->addr_space()) {
                case AddrSpace::Function: storage_class = spv::StorageClassFunction;     break;
                case AddrSpace::Private:  storage_class = spv::StorageClassPrivate;      break;
                case AddrSpace::Push:     storage_class = spv::StorageClassPushConstant; break;
                case AddrSpace::Global: {
                    storage_class = spv::StorageClassPhysicalStorageBuffer;
                    converted->datatype = std::make_unique<PtrDatatype>(converted);
                    break;
                }
                case AddrSpace::Generic: {
                    world().WLOG("Passing a generic pointer to a SPIR-V module. SpirV doesn't know about these, and so this will be passed as a 64 bit integer. Tread carefully !");
                    ConvertedType* conv_u64 = convert(world().type_pu64());
                    converted->type_id = conv_u64->type_id;
                    goto ptr_done;
                }
                default:
                    assert(false && "This address space is not supported");
                    break;
            }
            {
                const Type* pointee = ptr->pointee();
                while (auto arr = pointee->isa<IndefiniteArrayType>())
                    pointee = arr->elem_type();
                ConvertedType* element = convert(pointee);
                converted->type_id = builder_->declare_ptr_type(storage_class, element->type_id);
            }
            ptr_done:
            break;
        }
        case Node_IndefiniteArrayType: {
            world().ELOG("Using indefinite types directly is not permitted - they may only be pointed to");
            std::abort();
        }
        case Node_DefiniteArrayType: {
            auto array = type->as<DefiniteArrayType>();
            ConvertedType* element = convert(array->elem_type());
            SpvId size = builder_->constant(convert(world().type_pu32())->type_id, {(uint32_t) array->dim() });
            converted->type_id = builder_->declare_array_type(element->type_id, size);
            converted->datatype = std::make_unique<DefiniteArrayDatatype>(converted, element, array->dim());
            break;
        }

        case Node_ClosureType:
        case Node_FnType: {
            // extract "return" type, collect all other types
            auto fn = type->as<FnType>();
            ConvertedType* ret = nullptr;
            std::vector<SpvId> ops;
            for (auto op : fn->ops()) {
                if (op->isa<MemType>() || op == world().unit()) continue;
                auto fn_type = op->isa<FnType>();
                if (fn_type && !op->isa<ClosureType>()) {
                    assert(!ret && "only one 'return' supported");
                    std::vector<ConvertedType*> ret_types;
                    for (auto fn_op : fn_type->ops()) {
                        if (fn_op->isa<MemType>() || fn_op == world().unit()) continue;
                        ret_types.push_back(convert(fn_op));
                    }
                    if (ret_types.empty())          ret = convert(world().tuple_type({}));
                    else if (ret_types.size() == 1) ret = ret_types.back();
                    else                            assert(false && "Didn't we refactor this out yet by making functions single-argument ?");
                } else
                    ops.push_back(convert(op)->type_id);
            }
            assert(ret);

            if (type->tag() == Node_FnType) {
                converted->type_id = builder_->declare_fn_type(ops, ret->type_id);
            } else {
                assert(false && "TODO: handle closure mess");
                THORIN_UNREACHABLE;
            }
            break;
        }

        case Node_StructType:
        case Node_TupleType: {
            std::vector<ConvertedType*> types;
            std::vector<SpvId> spv_types;
            size_t total_serialized_size = 0;
            for (auto member_type : type->ops()) {
                if (member_type == world().unit() || member_type == world().mem_type()) continue;
                auto converted_member_type = convert(member_type);
                types.push_back(converted_member_type);
                spv_types.push_back(converted_member_type->type_id);
                total_serialized_size = converted_member_type->datatype->serialized_size();
            }
            if (total_serialized_size == 0) {
                outf("this one is void");
                converted->type_id = builder_->void_type;
                break;
            }

            converted->type_id = builder_->declare_struct_type(spv_types);
            builder_->name(converted->type_id, type->to_string());
            converted->datatype = std::make_unique<ProductDatatype>(converted, std::move(types));
            break;
        }

        case Node_VariantType: {
            assert(type->num_ops() > 0 && "empty variants not supported");
            auto tag_type = world().type_pu32();
            ConvertedType* converted_tag_type = convert(tag_type);

            size_t max_serialized_size = 0;
            for (auto member_type : type->as<VariantType>()->ops()) {
                if (member_type == world().unit() || member_type == world().mem_type()) continue;
                auto converted_member_type = convert(member_type);
                if (converted_member_type->datatype->serialized_size() > max_serialized_size)
                    max_serialized_size = converted_member_type->datatype->serialized_size();
            }

            if (max_serialized_size > 0) {
                auto payload_type = world().definite_array_type(world().type_pu32(), max_serialized_size);
                auto* converted_payload_type = convert(payload_type);

                std::vector<SpvId> spv_pair = {converted_tag_type->type_id, converted_payload_type->type_id};
                converted->type_id = builder_->declare_struct_type(spv_pair);
                converted->datatype = std::make_unique<ProductDatatype>(converted, std::vector<ConvertedType*>{ converted_tag_type, converted_payload_type });
            } else {
                // We keep this useless level of struct so the rest of the code doesn't need a special path to extract the tag
                std::vector<SpvId> spv_singleton = { converted_tag_type->type_id };
                converted->type_id = builder_->declare_struct_type(spv_singleton);
                converted->datatype = std::make_unique<ProductDatatype>(converted, std::vector<ConvertedType*>{ converted_tag_type });
            }
            builder_->name(converted->type_id, type->to_string());
            break;
        }

        case Node_MemType: {
            assert(false && "MemType cannot be converted to SPIR-V");
        }

        default:
            THORIN_UNREACHABLE;
    }

    return converted;
}

}