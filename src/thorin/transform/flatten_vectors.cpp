#include "thorin/transform/flatten_vectors.h"
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

        const NominalType *new_nominal_type = nullptr;
        if (element_type->isa<StructType>()) {
            new_nominal_type = world.struct_type(nominal_type->name() + "_flat", element_types.size());
        } else if (element_type->isa<VariantType>()) {
            new_nominal_type = world.variant_vector_type(nominal_type->name() + "_flat", element_types.size(), vector_length);
        } else {
            element_type->dump();
            vector_type->dump();
            THORIN_UNREACHABLE;
        }
        assert(new_nominal_type);
        for (size_t i = 0; i < element_types.size(); ++i)
            new_nominal_type->set(i, element_types[i]);
        return new_nominal_type;
    } else if (auto tuple_type = element_type->isa<TupleType>(); tuple_type && tuple_type->num_ops() == 0) {
        newtype = flatten_type(tuple_type);
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
    } else if (auto vecvariant = type->isa<VariantVectorType>()) {
        auto vector_length = vecvariant->length();
        auto newstruct = world.struct_type(vecvariant->name() + "_flat", vecvariant->num_ops() + 1);
        for (size_t i = 0; i < vecvariant->num_ops(); ++i) {
            auto op = vecvariant->op(i);
            auto result = flatten_type(world.vec_type(op, vector_length));
            assert(result && !result->isa<VectorExtendedType>());
            newstruct->set(i, result);
        }
        auto index_type = world.type_qu64(vector_length);
        newstruct->set(vecvariant->num_ops(), index_type);
        return newstruct;
    } else if (auto tuple = type->isa<TupleType>()) {
        std::vector<const Type*> element_types;
        bool changed = false;
        for (auto old_element : tuple->ops()) {
            auto result = flatten_type(old_element);
            changed |= (result != old_element);
            assert(result && !result->isa<VectorExtendedType>());
            element_types.emplace_back(result);
        }
        if (changed) {
            return world.tuple_type(element_types);
        } else {
            return tuple;
        }
    } else
        //The type returned here should either be a basic vector type, or should have no vector characteristics alltogether.
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
    } else if (auto cont = def->isa<Continuation>(); cont && (cont->is_intrinsic() || cont->empty())) {
        std::cerr << "Intrinsic: " << def->to_string() << "\n";
        if(cont == world.branch())
                return cont;
        Debug de = cont->debug();
        if (de.name == "predicated") {
            auto new_type = flatten_fn_type(cont->type());
            return world.continuation(new_type, cont->attributes(), de);
        /*} else if (de.name == "llvm.exp.f32" ||
                de.name == "llvm.exp.f64" ||
                de.name == "llvm.sqrt.f32" ||
                de.name == "llvm.sqrt.f64" ||
                de.name == "llvm.sin.f32" ||
                de.name == "llvm.sin.f64" ||
                de.name == "llvm.cos.f32" ||
                de.name == "llvm.cos.f64" ||
                de.name == "llvm.minnum.f32" ||
                de.name == "llvm.minnum.f64" ||
                de.name == "llvm.floor.f32" ||
                de.name == "llvm.floor.f64" ||
                de.name == "llvm.exp.v8f32" ||
                de.name == "llvm.exp.v8f64" ||
                de.name == "llvm.sqrt.v8f32" ||
                de.name == "llvm.sqrt.v8f64" ||
                de.name == "llvm.sin.v8f32" ||
                de.name == "llvm.sin.v8f64" ||
                de.name == "llvm.cos.v8f32" ||
                de.name == "llvm.cos.v8f64" ||
                de.name == "llvm.minnum.v8f32" ||
                de.name == "llvm.minnum.v8f64" ||
                de.name == "llvm.floor.v8f32" ||
                de.name == "llvm.floor.v8f64") {*/
        } else {
            std::cerr << "Unknown intrinsic " << de.name << "\n";
            auto new_type = flatten_fn_type(cont->type());
            return world.continuation(new_type, cont->attributes(), de);
        }
    } else if (auto cont = def->isa<Continuation>(); cont && !cont->is_intrinsic()) {
        auto new_continuation = flatten_continuation(cont, world);
        //assert(false && "return continuation");
        return new_continuation;
    } else if (def->isa<Param>()) {
        assert(false && "Parameters should be handled beforehand!");
    } else {
        std::cerr << "TODO: " << def->to_string() << ": " << def->type()->to_string() << "\n";
        assert(false && "TODO");
    }
}

static const PrimOp * flatten_primop(const PrimOp *primop) {
    World &world = *world_;
    
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
    } else if (auto load = primop->isa<Load>()) {
        if (newtype->like(load->type()))
            new_primop = primop->rebuild(nops, newtype)->as<PrimOp>();
        else {
            auto addresses = nops[1];
            auto mem = nops[0];
            assert(addresses->type()->isa<PtrType>());
            auto pointee_type = addresses->type()->as<PtrType>()->pointee();
            auto vector_width = addresses->type()->as<PtrType>()->length();

            assert(vector_width > 1);

            if (vector_width > 1)
                pointee_type = world.vec_type(pointee_type, vector_width);

            auto newstruct_type = newtype->op(1)->isa<StructType>();
            assert(newstruct_type);


            /* Better option for future consideration.
             * Does not work with variants currently, as variants dont support lea-ing into them.
            std::vector<const Def*> innerelements;
            for (size_t element_index = 0; element_index < values[0]->type()->num_ops(); element_index++) {
                auto index_splat = world.splat(world.literal_qs32(element_index));
                auto inner_lea = world.lea(addresses, index_spalt);
                auto load_element = world.load(mem, inner_lea);
                innerelements.emplace_back(load_element);
            }
            auto result_struct = world.struct_agg(newstruct_type, innerelements);
            */

            std::vector<const Def*> struct_elements;

            const Def* values[vector_width];

            for (size_t lane = 0; lane < vector_width; lane++) {
                auto ext = world.extract(addresses, lane);
                auto load = world.load(mem, ext)->as<Load>();
                mem = load->out_mem();
                values[lane] = load->out_val();
            }

            if (primop->op(1)->type()->isa<PtrType>() && primop->op(1)->type()->as<PtrType>()->pointee()->isa<VariantType>()) {
                for (size_t element_index = 0; element_index < values[0]->type()->num_ops(); element_index++) {
                    auto element = newstruct_type->op(element_index);
                    if (!element->isa<TupleType>() || element->as<TupleType>()->num_ops()) { //TODO: This is a hack to resolve issues with []
                        std::vector<const Def*> newelements;
                        for (size_t lane = 0; lane < vector_width; lane++) {
                            auto extract = world.variant_extract(values[lane], element_index);
                            newelements.emplace_back(extract);
                        }

                        if (element->isa<VectorType>()) {
                            auto newelement = world.vector(newelements); // This might still not work!
                            struct_elements.emplace_back(newelement);
                        } else if (auto structtype = element->isa<StructType>()) {
                            std::vector<const Def*> innerelements;
                            for (size_t element_index = 0; element_index < structtype->num_ops(); element_index++) {
                                std::vector<const Def*> elements;
                                for (size_t lane = 0; lane < vector_width; ++lane) {
                                    auto element = newelements[lane];
                                    auto extract = world.extract(element, element_index);
                                    elements.emplace_back(extract);
                                }

                                auto newelement = world.vector(elements);
                                innerelements.emplace_back(newelement);
                            }
                            struct_elements.emplace_back(world.struct_agg(structtype, innerelements));
                        } else {
                            THORIN_UNREACHABLE;
                        }
                    } else {
                        auto extract = world.variant_extract(values[0], element_index);
                        struct_elements.emplace_back(extract);
                    }
                }
                {
                    std::vector<const Def*> newelements;
                    for (size_t lane = 0; lane < vector_width; lane++) {
                        auto extract = world.variant_index(values[lane]);
                        newelements.emplace_back(extract);
                    }

                    auto newelement = world.vector(newelements); // This might still not work!
                    struct_elements.emplace_back(newelement);
                }
            } else {
                for (size_t element_index = 0; element_index < values[0]->type()->num_ops(); element_index++) {
                    std::vector<const Def*> newelements;
                    for (size_t lane = 0; lane < vector_width; lane++) {
                        auto extract = world.extract(values[lane], element_index);
                        newelements.emplace_back(extract);
                    }
                    auto newelement = world.vector(newelements); // This might still not work!
                    struct_elements.emplace_back(newelement);
                }
            }

            auto returnstruct = world.struct_agg(newstruct_type, struct_elements);
            auto return_tuple = world.tuple({mem, returnstruct})->as<PrimOp>();
            new_primop = return_tuple;
        }
    } else if (auto var_index = primop->isa<VariantIndex>()) {
        auto result_struct = nops[0];
        if (result_struct->type()->isa<StructType>()) {
            auto struct_length = result_struct->num_ops();
            new_primop = world.extract(result_struct, struct_length - 1)->as<PrimOp>();
        } else {
            new_primop = primop->rebuild(nops, newtype)->as<PrimOp>();
        }
    } else if (auto extract = primop->isa<Extract>()) {
        if (nops[1]->isa<Tuple>()) {
            assert(nops[1]->op(0)->isa<Top>());
            nops[1] = nops[1]->op(1);
        }
        new_primop = primop->rebuild(nops, newtype)->as<PrimOp>();
    } else {
        new_primop = primop->rebuild(nops, newtype)->as<PrimOp>();
    }

    assert(new_primop);
    if (!new_primop->type()->like(newtype)) {
        std::cerr << "Error\n";
        new_primop->dump();
        new_primop->type()->dump();
        newtype->dump();
        primop->type()->dump();

        std::cerr << primop->op_name() << " " << new_primop->op_name() << "\n";
    }
    assert(new_primop->type()->like(newtype));

    std::cerr << "Mapping " << primop->to_string() << " to " << new_primop->to_string() << "\n";
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

const Continuation* flatten_continuation(const Continuation* kernel, World& world) {
    //TODO: !!!
    world_ = &world;

    std::cerr << "Flatten Continuation " << kernel->to_string() << "\n";
    Continuation *orig = const_cast<Continuation*>(kernel);
    DUMP_BLOCK(orig);

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

    def2def[kernel] = ncontinuation;
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
    //std::cerr << "orig\n";
    //world.dump();

    std::cerr << "flattening\n";
    for (auto continuation : world.copy_continuations()) {
        if (continuation->num_params() > 1 && continuation->param(1)->type()->isa<VectorType>() && continuation->param(1)->type()->as<VectorType>()->is_vector()) {
            if (continuation->empty())
                continue;
            //continuation->dump();
            const Continuation* new_continuation = flatten_continuation(continuation, world);
            //std::cerr << "new block\n";
            Continuation *newb = const_cast<Continuation*>(new_continuation);
            //DUMP_BLOCK(newb);
            continuation->replace(newb);
        }
    }

    world.cleanup();

    //std::cerr << "Done flattening\n";
    //world.dump();

    //TODO: fix parameters and continuations
}

}
