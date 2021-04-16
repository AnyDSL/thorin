#include "thorin/be/spirv/spirv.h"

namespace thorin::spirv {

ScalarDatatype::ScalarDatatype(ConvertedType* type, int type_tag, size_t size_in_bytes, size_t alignment_in_bytes)
: Datatype(type), type_tag(type_tag), size_in_bytes(size_in_bytes), alignment(alignment_in_bytes)
{
    /// currently limited to 32-bit
    assert(size_in_bytes == 4);
}

SpvId ScalarDatatype::emit_deserialization(BasicBlockBuilder& bb, SpvId input) {
    auto loaded = bb.load(type->type_id, input);
    return bb.bitcast(type->type_id, loaded);
}

void ScalarDatatype::emit_serialization(BasicBlockBuilder& bb, SpvId output, SpvId data) {
    SpvId u32_tid = type->code_gen->convert(type->code_gen->world().type_pu32())->type_id;
    auto casted = bb.bitcast(u32_tid, data);
    bb.store(casted, output);
}

DefiniteArrayDatatype::DefiniteArrayDatatype(ConvertedType* type, ConvertedType* element_type, size_t length) : Datatype(type), element_type(element_type), length(length) {
    assert(element_type->datatype.get() != nullptr);
}

SpvId DefiniteArrayDatatype::emit_deserialization(BasicBlockBuilder& bb, SpvId input) {
    SpvId i32_tid = type->code_gen->convert(type->code_gen->world().type_ps32())->type_id;
    std::vector<SpvId> indices;
    std::vector<SpvId> elements;
    for (size_t i = 0; i < length; i++) {
        SpvId element_ptr = bb.ptr_access_chain(element_type->type_id, input, bb.file_builder.constant(i32_tid, { (uint32_t) (i * element_type->datatype->serialized_size()) }), indices);
        SpvId element = element_type->datatype->emit_deserialization(bb, element_ptr);
        elements.push_back(element);
    }
    return bb.composite(type->type_id, elements);
}
void DefiniteArrayDatatype::emit_serialization(BasicBlockBuilder& bb, SpvId output, SpvId data) {
    std::vector<SpvId> indices;
    SpvId i32_tid = type->code_gen->convert(type->code_gen->world().type_ps32())->type_id;
    for (size_t i = 0; i < length; i++) {
        SpvId element_ptr = bb.ptr_access_chain(element_type->type_id, output, bb.file_builder.constant(i32_tid, { (uint32_t) (i * element_type->datatype->serialized_size()) }), indices);
        element_type->datatype->emit_serialization(bb, element_ptr, bb.extract(element_type->type_id, data, { (uint32_t) i }));
    }
}

ProductDatatype::ProductDatatype(ConvertedType* type, const std::vector<ConvertedType*>&& elements_types) : Datatype(type), elements_types(elements_types) {
    for (auto& element_type : elements_types) {
        total_size += element_type->datatype->serialized_size();
    }
}

SpvId ProductDatatype::emit_deserialization(BasicBlockBuilder& bb, SpvId input) {
    SpvId i32_tid = type->code_gen->convert(type->code_gen->world().type_pu32())->type_id;
    std::vector<SpvId> indices;
    std::vector<SpvId> elements;
    size_t offset = 0;
    for (auto& element_type : elements_types) {
        SpvId element_ptr = bb.ptr_access_chain(element_type->type_id, input, bb.file_builder.constant(i32_tid, { (uint32_t) offset }), indices);
        SpvId element = element_type->datatype->emit_deserialization(bb, element_ptr);
        offset += element_type->datatype->serialized_size();
        elements.push_back(element);
    }
    return bb.composite(type->type_id, elements);
}
void ProductDatatype::emit_serialization(BasicBlockBuilder& bb, SpvId output, SpvId data) {
    SpvId i32_tid = type->code_gen->convert(type->code_gen->world().type_pu32())->type_id;
    std::vector<SpvId> indices;
    size_t offset = 0;
    int i = 0;
    for (auto& element_type : elements_types) {
        SpvId element_ptr = bb.ptr_access_chain(element_type->type_id, output, bb.file_builder.constant(i32_tid, { (uint32_t) offset }), indices);
        element_type->datatype->emit_serialization(bb, element_ptr, bb.extract(element_type->type_id, data, { (uint32_t) i++ }));
    }
}

ConvertedType* CodeGen::convert(const Type* type) {
    if (auto iter = types_.find(type); iter != types_.end()) return iter->second.get();

    assert(!type->isa<MemType>());
    ConvertedType* converted = types_.emplace(type, std::make_unique<ConvertedType>(this) ).first->second.get();
    switch (type->tag()) {
        // Boolean types are typically packed intelligently when declaring in local variables, however with vanilla Vulkan 1.0 they can only be represented via 32-bit integers
        // Using extensions, we could use 16 or 8-bit ints instead
        // We can also pack them inside structures using bit-twiddling tricks, if the need arises
        case PrimType_bool:
            converted->type_id = builder_->declare_bool_type();
            converted->datatype = std::make_unique<ScalarDatatype>(converted, type->tag(), 4, 4);
            break;
        case PrimType_ps8:  case PrimType_qs8:  case PrimType_pu8:  case PrimType_qu8:  assert(false && "TODO: look into capabilities to enable this");
        case PrimType_ps16: case PrimType_qs16: case PrimType_pu16: case PrimType_qu16: assert(false && "TODO: look into capabilities to enable this");
        case PrimType_ps32: case PrimType_qs32:
            converted->type_id = builder_->declare_int_type(32, true );
            converted->datatype = std::make_unique<ScalarDatatype>(converted, type->tag(), 4, 4);
            break;
        case PrimType_pu32: case PrimType_qu32:
            converted->type_id = builder_->declare_int_type(32, false);
            converted->datatype = std::make_unique<ScalarDatatype>(converted, type->tag(), 4, 4);
            break;
        case PrimType_ps64: case PrimType_qs64: case PrimType_pu64: case PrimType_qu64: assert(false && "TODO: look into capabilities to enable this");
        case PrimType_pf16: case PrimType_qf16:                                         assert(false && "TODO: look into capabilities to enable this");
        case PrimType_pf32: case PrimType_qf32:
            converted->type_id = builder_->declare_float_type(32);
            converted->datatype = std::make_unique<ScalarDatatype>(converted, type->tag(), 4, 4);
            break;
        case PrimType_pf64: case PrimType_qf64:                                         assert(false && "TODO: look into capabilities to enable this");
        case Node_PtrType: {
            auto ptr = type->as<PtrType>();
            spv::StorageClass storage_class;
            switch (ptr->addr_space()) {
                case AddrSpace::Function: storage_class = spv::StorageClassFunction; break;
                case AddrSpace::Private:  storage_class = spv::StorageClassPrivate;  break;
                default:
                    assert(false && "This address space is not supported");
                    break;
            }
            ConvertedType* element = convert(ptr->pointee());
            converted->type_id = builder_->declare_ptr_type(storage_class, element->type_id);
            break;
        }
        case Node_IndefiniteArrayType: {
            assert(false && "TODO");
            // auto array = type->as<IndefiniteArrayType>();
            // return types_[type] = spv_type;
            THORIN_UNREACHABLE;
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

        case Node_StructType: {
            std::vector<ConvertedType*> types;
            std::vector<SpvId> spv_types;
            for (auto elem : type->as<StructType>()->ops()) {
                auto member_type = convert(elem);
                types.push_back(member_type);
                spv_types.push_back(member_type->type_id);
            }
            converted->type_id = builder_->declare_struct_type(spv_types);
            builder_->name(converted->type_id, type->to_string());
            converted->datatype = std::make_unique<ProductDatatype>(converted, std::move(types));
            break;
        }

        case Node_TupleType: {
            std::vector<ConvertedType*> types;
            std::vector<SpvId> spv_types;
            for (auto elem : type->as<TupleType>()->ops()){
                auto member_type = convert(elem);
                types.push_back(member_type);
                spv_types.push_back(member_type->type_id);
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
            //std::vector<ConvertedType*> types;
            //std::vector<SpvId> spv_types;
            for (auto elem : type->as<VariantType>()->ops()){
                auto member_type = convert(elem);

                if (member_type->datatype->serialized_size() > max_serialized_size)
                    max_serialized_size = member_type->datatype->serialized_size();
            }

            auto payload_type = world().definite_array_type(world().type_pu32(), max_serialized_size);
            auto* converted_payload_type = convert(payload_type);

            std::vector<SpvId> spv_pair = {converted_tag_type->type_id, converted_payload_type->type_id };
            converted->type_id = builder_->declare_struct_type(spv_pair);

            // auto oh_god_why = std::vector<ConvertedType*> ( &converted_tag_type, &converted_payload_type );

            converted->datatype = std::make_unique<ProductDatatype>(converted, std::vector<ConvertedType*> { converted_tag_type, converted_payload_type });
            builder_->name(converted->type_id, type->to_string());
            break;
        }

        default:
            THORIN_UNREACHABLE;
    }

    return converted;
}

}