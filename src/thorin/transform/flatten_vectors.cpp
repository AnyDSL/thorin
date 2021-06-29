#include "thorin/continuation.h"
#include "thorin/world.h"
#include "thorin/transform/mangle.h"
#include "thorin/analyses/verify.h"
#include "thorin/analyses/schedule.h"

#include <limits>

#define DUMP_BLOCK(block) { \
                    Stream s(std::cout); \
                    RecStreamer rec(s, std::numeric_limits<size_t>::max()); \
                    for (auto& block : schedule(Scope(block))) { \
                        rec.conts.push(block); \
                        rec.run(); \
                    } \
                    s.endl(); \
}

namespace thorin {

    //class actually defined in rec_stream.cpp!
class RecStreamer {
public:
    RecStreamer(Stream& s, size_t max)
        : s(s)
        , max(max)
    {}

    void run();
    void run(const Def*);

    Stream& s;
    size_t max;
    unique_queue<ContinuationSet> conts;
    DefSet defs;
};

//TODO: !!!
static World* world_;

Def2Def def2def;

static const PrimOp * flatten_primop(const PrimOp *primop);
static const Type* flatten_type(const Type *type);
static const Type* flatten_vector_type(const VectorExtendedType *vector_type);
static const Continuation* flatten_continuation(const Continuation* kernel);

static const Type* flatten_vector_type(const VectorExtendedType *vector_type) {
    World &world = *world_;

    auto vector_length = vector_type->length();
    auto element_type = vector_type->element();

    const Type* newtype = nullptr;
    if (auto element_vector_type = element_type->isa<PrimType>()) {
        assert(element_vector_type->length() == 1);
        newtype = world.prim_type(element_vector_type->primtype_tag(), vector_length);
    } else if (auto element_vector_type = element_type->isa<PtrType>()) {
        assert(element_vector_type->length() == 1);
        newtype = world.ptr_type(element_vector_type->pointee(), vector_length, element_vector_type->device(), element_vector_type->addr_space());
    } else if (auto nominal_type = element_type->isa<NominalType>()) {
        std::vector<const Type*> element_types;
        for (auto old_element : nominal_type->ops()) {
            const Type *result = nullptr;

            if (auto vectype = old_element->isa<VectorType>(); vectype && vectype->is_vector()) {
                assert(false && "No support for stacked vectors at this point.");
            } else if (auto primtype = old_element->isa<PrimType>()) {
                result = world.prim_type(primtype->primtype_tag(), vector_length);
            } else if (auto ptrtype = old_element->isa<PtrType>()) {
                result = world.ptr_type(ptrtype->pointee(), vector_length, ptrtype->device(), ptrtype->addr_space());
            } else {
                result = flatten_vector_type(world.vec_type(old_element, vector_length)->as<VectorExtendedType>());
            }
            assert(result);
            assert(!result->isa<VectorExtendedType>());
            
            element_types.emplace_back(result);
        }

        //for (auto element : nominal_type->ops())
            //element->dump();
        for (auto element : element_types)
            element->dump();

        const NominalType *new_nominal_type;
        if (element_type->isa<StructType>()) {
            new_nominal_type = world.struct_type(nominal_type->name() + "_flat", element_types.size());
        } else if (element_type->isa<VariantType>()) {
            new_nominal_type = world.variant_vector_type(nominal_type->name() + "_flat", element_types.size(), vector_length);
        }
        assert(new_nominal_type);
        for (size_t i = 0; i < element_types.size(); ++i)
            new_nominal_type->set(i, element_types[i]);
        return new_nominal_type;
    } else if (auto tuple_type = element_type->isa<TupleType>(); tuple_type && tuple_type->num_ops() == 0) {
        return element_type;
    } else {
        element_type->dump();
        THORIN_UNREACHABLE;
    }

    return newtype;
}

static const Type* flatten_type(const Type *type) {
    World &world = *world_;

    if (auto vector = type->isa<VectorExtendedType>()) {
        return flatten_vector_type(vector);
    } else if (auto variant = type->isa<VariantVectorType>()) {
        auto result_variant = world.variant_vector_type(variant->name() + "_flat", variant->num_ops(), variant->length());
        for (size_t i = 0; i < variant->num_ops(); ++i) {
            auto old_element = variant->op(i);
            auto result = flatten_type(old_element);
            assert(result && !result->isa<VectorExtendedType>());
            result_variant->set(i, result);
        }
        return result_variant;
    } else if (auto tuple = type->isa<TupleType>()) {
        std::vector<const Type*> element_types;
        for (auto old_element : tuple->ops()) {
            auto result = flatten_type(old_element);
            assert(result && !result->isa<VectorExtendedType>());
            element_types.emplace_back(result);
        }
        return world.tuple_type(element_types);
    } else if (auto ptr = type->isa<PtrType>()) {
        if (!ptr->is_vector())
            return type;
        else {
            auto pointee = ptr->pointee();
            auto flat_element = flatten_type(world.vec_type(pointee, ptr->length()));
            return world.ptr_type(flat_element, 1, ptr->device(), ptr->addr_space());
        }
    } else
        return type;
}


static const FnType* flatten_fn_type(const FnType *fntype) {
    World &world = *world_;

    std::vector<const Type*> arg_types;
    for (auto op : fntype->ops()) {
        const Type* result = nullptr;
        if (auto vecextended = op->isa<VectorExtendedType>())
            result = flatten_type(vecextended);
        else if (auto fn = op->isa<FnType>())
            result = flatten_fn_type(fn);
        else
            result = op;
        arg_types.emplace_back(result);
    }
    return world.fn_type(arg_types);
}

static const Def * flatten_def(const Def *def) {
    World &world = *world_;
    
    std::cerr << "flatten def "  << def->to_string() << "\n";

    auto replacement = def2def[def];
    if (replacement)
        return replacement;

    if (auto primop = def->isa<PrimOp>()) {
        return flatten_primop(primop);
    } else if (auto cont = def->isa<Continuation>(); cont && cont->is_intrinsic()) {
        std::cerr << "Intrinsic: " << def->to_string() << "\n";
        Debug de = cont->debug();
        if (de.name == "predicated") {
            auto new_type = flatten_fn_type(cont->type());
            cont->type()->dump();
            new_type->dump();
            return world.continuation(new_type, cont->attributes(), de);
        } else {
            std::cerr << "Unknown intrinsic\n";
            THORIN_UNREACHABLE;
        }
    } else if (auto cont = def->isa<Continuation>(); cont && !cont->is_intrinsic()) {
        auto new_continuation = flatten_continuation(cont);
        //assert(false && "return continuation");
        return new_continuation;
    } else {
        std::cerr << "TODO: " << def->to_string() << "\n";
        assert(false && "TODO");
    }
}

static const PrimOp * flatten_primop(const PrimOp *primop) {
    //World &world = *world_;
    
    std::cerr << "flatten primop "  << primop->to_string() << "\n";

    auto replacement = def2def[primop];
    if (replacement)
        return replacement->as<PrimOp>();

    auto primop_type = primop->type();
    assert(primop_type);
    const Type* newtype;
    newtype = flatten_type(primop_type);
    assert(newtype);

    Array<const Def*> nops(primop->num_ops());

    for (size_t i = 0, e = primop->num_ops(); i != e; ++i) {
        nops[i] = flatten_def(primop->op(i));
    }

    const PrimOp* new_primop;

    if (primop->isa<PrimLit>()) {
        new_primop = primop;
    } else if (primop->isa<LEA>()) {
        std::cerr << "LEA\n";
        auto vec_in = nops[1]->type()->isa<VectorType>() || nops[1]->type()->isa<VariantVectorType>();
        auto vec_out = newtype->isa<VectorType>();
        if (vec_in && !vec_out) {
#if 0
            //During flattening, this can happen, if the pointer has to be deconstructed through multiple lea instructions.
            assert(newtype->isa<StructType>());
            int vector_length = vec_in->length();
            auto struct_type = newtype->as<StructType>();
            auto ptr = nops[0];
            std::vector<const Def*> leas;
            std::vector<std::vector<const Def*>> inner_leas;
            for (int i = 0; i < vector_length; ++i) {
                auto extract_element = world.extract(nops[1], world.literal(thorin::qs32{i}));
                auto lea = world.lea(ptr, extract_element, primop->debug());
                leas.emplace_back(lea);
                lea->dump();
                lea->type()->dump();
                std::vector<const Def*> emptyvector;
                inner_leas.emplace_back(emptyvector);
            }
            for (int element_index = 0; element_index < struct_type->num_ops(); ++element_index) {
                auto element = struct_type->op(element_index);
                element->dump();
                for (int i = 0; i < vector_length; ++i) {
                    auto element_lea = leas[i];
                    auto inner_lea = world.lea(element_lea, world.literal(thorin::qs32(i)), primop->debug());
                    inner_lea->dump();
                    inner_lea->type()->dump();
                    assert(inner_lea->type()->isa<PtrType>());
                    assert(inner_lea->type()->as<PtrType>()->pointee() == element);
                    inner_leas[element_index].emplace_back(inner_lea);
                }
            }

            //TODO: Instead of this approach, I should alter the behaviour of load and store instructions to deal with those LEA results propperly.

            std::vector<const Def*> newelements;
            for i in range(vector_length) {
                index = lea-vector[i];
                target_element = lea(ptr, index);
                for part in target_element {
                    newelements[part.index] = part;
                }
            }
#endif
            assert(false);
        } else {
            new_primop = primop->rebuild(nops, newtype)->as<PrimOp>();
            assert(false);
        }
    } else {
        new_primop = primop->rebuild(nops, newtype)->as<PrimOp>();
    }

    assert(new_primop);
    if(new_primop->type() != newtype) {
        std::cerr << "Error\n";
        new_primop->dump();
        new_primop->type()->dump();
        newtype->dump();

        std::cerr << new_primop->op_name() << "\n";
    }
    assert(new_primop->type() == newtype);

    def2def[primop] = new_primop;

    return new_primop;
}

void flatten_body(const Continuation *old_continuation, Continuation *new_continuation) {
    std::cerr << "Flattening " << old_continuation->to_string() << " into " << new_continuation->to_string() << "\n";
    assert(!old_continuation->empty());

    Array<const Def*>nops(old_continuation->num_ops());
    for (size_t i = 0, e = nops.size(); i != e; ++i)
        nops[i] = flatten_def(old_continuation->op(i));

    Array<const Def*> nargs(nops.size() - 1); //new args of new_continuation
    const Def* ntarget = nops.front();   // new target of new_continuation

    for (size_t i = 0; i < nops.size() - 1; i++)
        nargs[i] = nops[i + 1];

    new_continuation->jump(ntarget, nargs, old_continuation->debug());
}

static const Continuation* flatten_continuation(const Continuation* kernel) {
    World &world = *world_;
    Continuation *ncontinuation;

    std::vector<const Type*> param_types;
    for (size_t i = 0, e = kernel->num_params(); i != e; i++) {
        auto paramtype = kernel->param(i)->type();
        if (auto vector_type = paramtype->isa<VectorExtendedType>()) {
            auto newtype = flatten_type(vector_type);
            param_types.emplace_back(newtype);
        } else
            param_types.emplace_back(paramtype);
    }

    auto fn_type = world.fn_type(param_types);
    ncontinuation = world.continuation(fn_type, kernel->debug_history());

    def2def[kernel] = kernel;
    for (size_t i = 0, j = 0, e = kernel->num_params(); i != e; ++i) {
        auto old_param = kernel->param(i);
        auto new_param = ncontinuation->param(j++);
        assert(new_param);
        def2def[old_param] = new_param;
        new_param->debug().name = old_param->name();
    }

    // mangle filter
    if (!kernel->filter().empty()) {
        Array<const Def*> new_filter(ncontinuation->num_params());
        size_t j = 0;
        for (size_t i = 0, e = kernel->num_params(); i != e; ++i) {
            new_filter[j++] = flatten_def(kernel->filter(i));
        }

        for (size_t e = ncontinuation->num_params(); j != e; ++j)
            new_filter[j] = world.literal_bool(false, Debug{});

        ncontinuation->set_filter(new_filter);
    }

    flatten_body(kernel, ncontinuation);

    return ncontinuation;
}

void flatten_vectors(World& world) {
    //TODO: !!!
    world_ = &world;

    std::cerr << "orig\n";
    world.dump();

    std::cerr << "flattening\n";
    for (auto continuation : world.copy_continuations()) {
        if (continuation->num_params() > 1 && continuation->param(1)->type()->isa<VectorType>() && continuation->param(1)->type()->as<VectorType>()->is_vector()) {
            if (continuation->empty())
                continue;
            continuation->dump();
            const Continuation* new_continuation = flatten_continuation(continuation);
            std::cerr << "new block\n";
            Continuation *newb = const_cast<Continuation*>(new_continuation);
            DUMP_BLOCK(newb);
            //continuation.replace_calls(new_continuation);
        }
    }

    world.cleanup();

    std::cerr << "Done flattening\n";
    world.dump();

    //TODO: fix parameters and continuations
}

}
