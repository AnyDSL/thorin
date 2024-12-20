#include "spirv_private.h"
#include "thorin/util/stream.h"
#include "thorin/util/utility.h"

namespace thorin::spirv {

uint32_t CodeGen::convert(AddrSpace as) {
    spv::StorageClass storage_class;
    switch (as) {
        case AddrSpace::Function: storage_class = spv::StorageClassFunction;              break;
        case AddrSpace::Private: {
            storage_class = spv::StorageClassPrivate;
            if (target_info_.dialect != Target::Dialect::Vulkan)
                builder_->capability(spv::CapabilityVectorComputeINTEL);
            break;
        }
        case AddrSpace::Generic: {
            storage_class = spv::StorageClassGeneric;
            builder_->capability(spv::Capability::CapabilityGenericPointer);
            break;
        }
        case AddrSpace::Shared: {
            storage_class = spv::StorageClassWorkgroup;
            break;
        }
        case AddrSpace::Push:     storage_class = spv::StorageClassPushConstant;          break;
        case AddrSpace::Input:    storage_class = spv::StorageClassInput;                 break;
        case AddrSpace::Output:   storage_class = spv::StorageClassOutput;                break;
        case AddrSpace::Global: {
            if (target_info_.dialect == Target::Dialect::Vulkan) {
                builder_->capability(spv::Capability::CapabilityPhysicalStorageBufferAddresses);
                storage_class = spv::StorageClassPhysicalStorageBuffer;
            } else
                storage_class = spv::StorageClassCrossWorkgroup;
            break;
        }
        default:
            assert(false && "This address space is not supported");
            break;
    }
    return storage_class;
}

Id CodeGen::get_codom_type(const FnType* fn) {
    auto [dom, codom] = get_dom_codom(fn);
    assert(codom);
    return codom;
}

std::tuple<std::vector<Id>, Id> CodeGen::get_dom_codom(const FnType* fn) {
    Id ret = 0;
    std::vector<Id> ops;
    for (auto op : fn->types()) {
        auto fn_type = op->isa<FnType>();
        if (fn_type && !op->isa<ClosureType>()) {
            assert(!ret && "only one 'return' supported");
            std::vector<const Type*> ret_types;
            for (auto fn_op : fn_type->types()) {
                if (!should_emit(fn_op))
                    continue;
                ret_types.push_back(fn_op);
            }
            if (ret_types.size() == 1)
                ret = convert_maybe_void(ret_types.back()).id;
            else
                ret = convert_maybe_void(world().tuple_type(ret_types)).id;
        } else if (!should_emit(op))
            continue;
        else
            ops.push_back(convert(op).id);
    }
    return std::make_tuple(ops, ret);
}

ConvertedType CodeGen::convert_maybe_void(const thorin::Type* type) {
    auto converted = convert(type);

    if ((type->isa<StructType>() || type->isa<TupleType>()) && converted.layout->size == 0) {
        converted.id = builder_->declare_void_type();
        converted.layout = std::nullopt;
        return converted;
    }

    return converted;
}

ConvertedType CodeGen::convert(const Type* type) {
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

    // OpenCL has no signed integer types
    if (target_info_.dialect == Target::OpenCL) {
        switch (type->tag()) {
            case Node_PrimType_ps8:
                type = world().prim_type(PrimType_pu8, type->as<VectorType>()->length()); \
                break;
            case Node_PrimType_ps16:
                type = world().prim_type(PrimType_pu16, type->as<VectorType>()->length()); \
                break;
            case Node_PrimType_ps32:
                type = world().prim_type(PrimType_pu32, type->as<VectorType>()->length()); \
                break;
            case Node_PrimType_ps64:
                type = world().prim_type(PrimType_pu64, type->as<VectorType>()->length()); \
                break;
            default: break;
        }
    }

    if (auto iter = types_.find(type); iter != types_.end())
        return iter->second;

    // Vector types are stupid and dangerous!

    ConvertedType converted = { 0, std::nullopt };

    if (auto vec = type->isa<VectorType>(); vec && vec->length() > 1) {
        auto component = vec->scalarize();
        auto conv_comp = convert(component);
        converted.id = builder_->declare_vector_type(conv_comp.id, (uint32_t) vec->length());
        converted.layout = conv_comp.layout;
        converted.layout->size *= vec->length();
    } else switch (type->tag()) {
        // Boolean types are typically packed intelligently when declaring in local variables, however with vanilla Vulkan 1.0 they can only be represented via 32-bit integers
        // Using extensions, we could use 16 or 8-bit ints instead
        // We can also pack them inside structures using bit-twiddling tricks, if the need arises
        // Note: this only affects storing booleans inside structures, for regular variables the actual spir-v bool type is used.
        case Node_PrimType_bool:
            converted.id = builder_->declare_bool_type();
            converted.layout = { 1, 1 };
            break;
        case Node_PrimType_ps8:
            builder_->capability(spv::Capability::CapabilityInt8);
            converted.id = builder_->declare_int_type(8, true);
            converted.layout = { 1, 1 };
            break;
        case Node_PrimType_pu8:
            builder_->capability(spv::Capability::CapabilityInt8);
            converted.id = builder_->declare_int_type(8, false);
            converted.layout = { 1, 1 };
            break;
        case Node_PrimType_ps16:
            builder_->capability(spv::Capability::CapabilityInt16);
            converted.id = builder_->declare_int_type(16, true);
            converted.layout = { 2, 2 };
            break;
        case Node_PrimType_pu16:
            builder_->capability(spv::Capability::CapabilityInt16);
            converted.id = builder_->declare_int_type(16, false);
            converted.layout = { 2, 2 };
            break;
        case Node_PrimType_ps32:
            converted.id = builder_->declare_int_type(32, true );
            converted.layout = { 4, 4 };
            break;
        case Node_PrimType_pu32:
            converted.id = builder_->declare_int_type(32, false);
            converted.layout = { 4, 4 };
            break;
        case Node_PrimType_ps64:
            builder_->capability(spv::Capability::CapabilityInt64);
            converted.id = builder_->declare_int_type(64, true);
            converted.layout = { 8, 8 };
            break;
        case Node_PrimType_pu64:
            builder_->capability(spv::Capability::CapabilityInt64);
            converted.id = builder_->declare_int_type(64, false);
            converted.layout = { 8, 8 };
            break;
        case Node_PrimType_pf16:
            builder_->capability(spv::Capability::CapabilityFloat16);
            converted.id = builder_->declare_float_type(16);
            converted.layout = { 2, 2 };
            break;
        case Node_PrimType_pf32:
            converted.id = builder_->declare_float_type(32);
            converted.layout = { 4, 4 };
            break;
        case Node_PrimType_pf64:
            builder_->capability(spv::Capability::CapabilityFloat64);
            converted.id = builder_->declare_float_type(64);
            converted.layout = { 8, 8 };
            break;
        case Node_PtrType: {
            auto ptr = type->as<PtrType>();
            const Type* pointee = ptr->pointee();
            while (auto arr = pointee->isa<IndefiniteArrayType>())
                pointee = arr->elem_type();
            converted.id = builder_->declare_ptr_type(static_cast<spv::StorageClass>(convert(ptr->addr_space())), convert(pointee).id);
            converted.layout = { target_info_.mem_layout.pointer_size, target_info_.mem_layout.pointer_size };
            break;
        }
        case Node_IndefiniteArrayType: {
            world().ELOG("Using indefinite types directly is not permitted - they may only be pointed to");
            std::abort();
        }
        case Node_DefiniteArrayType: {
            auto array = type->as<DefiniteArrayType>();
            auto element = convert(array->elem_type());
            converted.id = builder_->declare_array_type(element.id, builder_->u32_constant(array->dim()));
            converted.layout = { element.layout->size * array->dim(), element.layout->alignment };
            break;
        }

        case Node_ClosureType:
        case Node_FnType: {
            auto [dom, codom] = get_dom_codom(type->as<FnType>());

            if (type->tag() == Node_FnType) {
                converted.id = builder_->declare_fn_type(dom, codom);
            } else {
                assert(false && "TODO: handle closure mess");
                THORIN_UNREACHABLE;
            }
            break;
        }

        case Node_StructType:
        case Node_TupleType: {
            std::vector<Id> spv_types;
            size_t total_serialized_size = 0;
            converted.layout = { 0, 0 };
            for (auto member : type->ops()) {
                auto member_type = member->as<Type>();
                if (member_type == world().unit_type() || member_type == world().mem_type() || member_type->isa<FrameType>()) continue;
                auto converted_member_type = convert(member_type);
                assert(converted_member_type.layout);
                spv_types.push_back(converted_member_type.id);
                converted.layout->alignment = std::max(converted.layout->alignment, converted_member_type.layout->alignment);
                converted.layout->size = pad(converted.layout->size + converted_member_type.layout->size, converted.layout->alignment);
            }

            converted.id = builder_->declare_struct_type(spv_types);
            builder_->name(converted.id, type->to_string());
            break;
        }

        case Node_VariantType: {
            assert(type->num_ops() > 0 && "empty variants not supported");
            auto tag_type = world().type_pu32();

            size_t max_serialized_size = 0;
            for (auto member : type->as<VariantType>()->types()) {
                auto member_type = member->as<Type>();
                if (member_type == world().unit_type() || member_type == world().mem_type()) continue;
                auto converted_member_type = convert(member_type);
                assert(converted_member_type.layout);
                if (converted_member_type.layout->size > max_serialized_size)
                    max_serialized_size = converted_member_type.layout->size;
            }

            if (max_serialized_size > 0) {
                auto payload_type = world().definite_array_type(world().type_pu8(), max_serialized_size);
                auto struct_t = world().struct_type(type->name(), 2);
                struct_t->set_op(0, tag_type);
                struct_t->set_op(1, payload_type);
                converted = convert(struct_t);
                converted.variant.payload_t = std::make_optional(payload_type);
            } else {
                // We keep this useless level of struct so the rest of the code doesn't need a special path to extract the tag
                auto struct_t = world().struct_type(type->name(), 1);
                struct_t->set_op(0, tag_type);
                converted = convert(struct_t);
            }
            break;
        }

        case Node_MemType: {
            assert(false && "MemType cannot be converted to SPIR-V");
        }

        default:
            THORIN_UNREACHABLE;
    }

    types_[type] = converted;
    return converted;
}

}
