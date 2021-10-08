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
                    for (auto& block : schedule(Scope(const_cast<Continuation*>(block)))) { \
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

class Flatten {
    Def2Def def2def;
    Type2Type type2type;
    World& world;

    const Type* flatten_type(const Type*);
    const Type* flatten_vector_type(const VectorExtendedType*);
    const FnType* flatten_fn_type(const FnType *);

    const PrimOp* flatten_primop(const PrimOp*);
    const Def * flatten_def(const Def *);
    const Continuation* flatten_continuation(const Continuation*);
    void flatten_body(const Continuation *, Continuation *);

public:
    Flatten(World &world) : world(world) {};
    bool run();
};

const Type* Flatten::flatten_vector_type(const VectorExtendedType *vector_type) {
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
        const StructType* aggregation_type = world.struct_type(nominal_type->name() + "_flat", vector_length);
        for (size_t i = 0; i < vector_length; ++i) {
            aggregation_type->set(i, nominal_type);
        }
        return aggregation_type;
    } else if (auto tuple_type = element_type->isa<TupleType>(); tuple_type && tuple_type->num_ops() == 0) {
        newtype = flatten_type(tuple_type);
    } else {
        element_type->dump();
        THORIN_UNREACHABLE;
    }

    return newtype;
}

const Type* Flatten::flatten_type(const Type *type) {
    auto flattened = type2type[type];
    if (flattened)
        return flattened;

    if (auto vector = type->isa<VectorExtendedType>()) {
        return type2type[type] = flatten_vector_type(vector);
    } else if (auto vecvariant = type->isa<VariantVectorType>()) {
        assert(false && "Currently not applicable");
        auto vector_length = vecvariant->length();
        auto varianttype = world.variant_type(vecvariant->name(), vecvariant->num_ops());
        for (size_t i = 0; i < vecvariant->num_ops(); ++i) {
            auto op = vecvariant->op(i);
            varianttype->set(i, op); //TODO: Find the old variant instead.
        }
        auto newstruct = world.struct_type(vecvariant->name() + "_flat", vector_length);
        for (size_t i = 0; i < vector_length; ++i) {
            newstruct->set(i, varianttype);
        }
        return type2type[type] = newstruct;
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
            return type2type[type] = world.tuple_type(element_types);
        } else {
            return type2type[type] = tuple;
        }
    } else
        //The type returned here should either be a basic vector type, or should have no vector characteristics alltogether.
        return type2type[type] = type;
}


const FnType* Flatten::flatten_fn_type(const FnType *fntype) {
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

const Def * Flatten::flatten_def(const Def *def) {
    //std::cerr << "flatten def "  << def->to_string() << "\n";

    auto replacement = def2def[def];
    if (replacement)
        return replacement;

    if (auto primop = def->isa<PrimOp>()) {
        return flatten_primop(primop);
    } else if (auto cont = def->isa<Continuation>(); cont && (cont->is_intrinsic() || cont->empty())) {
        //std::cerr << "Intrinsic: " << def->to_string() << "\n";
        if(cont == world.branch())
                return cont;
        Debug de = cont->debug();
        if (de.name == "predicated") {
            auto new_type = flatten_fn_type(cont->type());
            return world.continuation(new_type, cont->attributes(), de);
        } else {
            //std::cerr << "Unknown intrinsic " << de.name << "\n";
            auto new_type = flatten_fn_type(cont->type());
            return world.continuation(new_type, cont->attributes(), de);
        }
    } else if (auto cont = def->isa<Continuation>(); cont && !cont->is_intrinsic()) {
        auto new_continuation = flatten_continuation(cont);
        //assert(false && "return continuation");
        return new_continuation;
    } else if (def->isa<Param>()) {
        assert(false && "Parameters should be handled beforehand!");
    } else {
        std::cerr << "TODO: " << def->to_string() << ": " << def->type()->to_string() << "\n";
        assert(false && "TODO");
    }
}

const PrimOp * Flatten::flatten_primop(const PrimOp *primop) {
    //std::cerr << "flatten primop "  << primop->to_string() << "\n";

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
    } else if (auto store = primop->isa<Store>()) {
        if (store->val()->type() == nops[2]->type() &&
            store->ptr()->type() == nops[1]->type())
            new_primop = primop->rebuild(world, newtype, nops)->as<PrimOp>();
        else {
            auto mem = nops[0];
            auto addresses = nops[1];
            auto values = nops[2];
            
            //std::cerr << "addresses " << addresses->type()->to_string() << "\n";
            //std::cerr << "values " << values->type()->to_string() << "\n";

            assert(addresses->type()->isa<PtrType>());
            //auto pointee_type = addresses->type()->as<PtrType>()->pointee();
            auto vector_width = addresses->type()->as<PtrType>()->length();

            //std::cerr << "pointee_type " << pointee_type->to_string() << "\n";
            //std::cerr << "vector_width " << vector_width << "\n";

            const Store* lane_store = nullptr;

            for (size_t lane = 0; lane < vector_width; ++lane) {
                auto ext = world.extract(addresses, lane);
                auto value = world.extract(values, lane);

                //std::cerr << "value " << lane << " :" << value->type()->to_string() << "\n";
                //std::cerr << "address " << lane << " :" << ext->type()->to_string() << "\n";

                lane_store = world.store(mem, ext, value)->as<Store>();

                mem = lane_store->out_mem();
            }

            new_primop = lane_store;
        }
    } else if (auto load = primop->isa<Load>()) {
        if (newtype->like(load->type()))
            new_primop = primop->rebuild(world, newtype, nops)->as<PrimOp>();
        else {
            assert(newtype->isa<TupleType>());
            auto structtype = newtype->as<TupleType>()->op(1);

            auto addresses = nops[1];
            auto mem = nops[0];

            assert(addresses->type()->isa<PtrType>());

            auto pointee_type = addresses->type()->as<PtrType>()->pointee();
            auto vector_width = addresses->type()->as<PtrType>()->length();

            assert(vector_width > 1);
            pointee_type = world.vec_type(pointee_type, vector_width);

            std::vector<const Def*> values;
            for (size_t lane = 0; lane < vector_width; lane++) {
                auto ext = world.extract(addresses, lane);
                auto load = world.load(mem, ext)->as<Load>();
                mem = load->out_mem();
                values.emplace_back(load->out_val());
            }

            auto returnstruct = world.struct_agg(structtype, values);
            auto return_tuple = world.tuple({mem, returnstruct})->as<PrimOp>();

            new_primop = return_tuple;
        }
    } else if (primop->isa<VariantIndex>()) {
        auto result_struct = nops[0];
        if (result_struct->type()->isa<StructType>()) {
            auto vector_width = result_struct->type()->num_ops();
            std::vector<const Def*> elements;
            for (size_t lane = 0; lane < vector_width; ++lane) {
                auto element = world.extract(result_struct, lane);
                auto element_variant_index = world.variant_index(element);
                elements.emplace_back(element_variant_index);
            }
            new_primop = world.vector(elements)->as<PrimOp>();
        } else {
            new_primop = primop->rebuild(world, newtype, nops)->as<PrimOp>();
        }
    } else if (auto var_extract = primop->isa<VariantExtract>()) {
        auto result_struct = nops[0];
        if (result_struct->type()->isa<StructType>()) {
            size_t variant_index = var_extract->index();
            auto vector_width = result_struct->type()->num_ops();
            std::vector<const Def*> elements;
            for (size_t lane = 0; lane < vector_width; ++lane) {
                auto element = world.extract(result_struct, lane);
                auto element_variant_extract = world.variant_extract(element, variant_index);
                elements.emplace_back(element_variant_extract);
            }
            new_primop = world.vector(elements)->as<PrimOp>(); //TODO: This might require additional lowering.
        } else {
            new_primop = primop->rebuild(world, newtype, nops)->as<PrimOp>();
        }
    } else if (primop->isa<Extract>()) {
        if (nops[1]->isa<Tuple>()) {
            assert(nops[1]->op(0)->isa<Top>());
            auto index = nops[1]->op(1);
            size_t vector_width = primop->op(0)->type()->as<VectorType>()->length();
            //std::cerr << "width " << vector_width << "\n";
            std::vector<const Def*> elements;
            //nops[0]->type()->dump();
            for (size_t i = 0; i < vector_width; ++i) {
                auto element = world.extract(nops[0], i);
                auto extract_result  = world.extract(element, index);
                elements.emplace_back(extract_result);
            }
            auto expected_type = world.vec_type(elements[0]->type(), vector_width);
            if (expected_type->like(newtype))
                new_primop = world.vector(elements)->as<PrimOp>();
            else
                new_primop = world.struct_agg(newtype, elements)->as<PrimOp>();
        } else {
            new_primop = primop->rebuild(world, newtype, nops)->as<PrimOp>();
        }
    } else if (auto agg = primop->isa<StructAgg>()) {
        if (newtype->like(agg->type())) {
            new_primop = primop->rebuild(world, newtype, nops)->as<PrimOp>();
        } else {
            auto element_type = newtype->op(0);
            size_t vector_width = primop->type()->as<VectorType>()->length();
            //element_type->dump();
            assert(element_type->isa<StructType>());

            std::vector<const Def*> rebuild_struct_elements;
            for (size_t lane = 0; lane < vector_width; ++lane) {
                std::vector<const Def*> inner_elements;
                for (size_t element_index = 0; element_index < element_type->num_ops(); ++element_index) {
                    auto element = nops[element_index];
                    auto inner_element = world.extract(element, lane);
                    inner_elements.emplace_back(inner_element);
                }
                auto lane_element = world.struct_agg(element_type, inner_elements);
                rebuild_struct_elements.emplace_back(lane_element);
            }
            new_primop = world.struct_agg(newtype, rebuild_struct_elements)->as<PrimOp>();
        }
    } else if (auto variant = primop->isa<Variant>()) {
        if (newtype->isa<StructType>()) {
            auto element_type = newtype->op(0);
            size_t vector_width = primop->type()->as<VectorType>()->length();
            std::vector<const Def*> rebuild_struct_elements;
            for (size_t lane = 0; lane < vector_width; ++lane) {
                auto inner_element = world.extract(nops[0], lane);
                auto lane_element = world.variant(element_type, inner_element, variant->index());
                rebuild_struct_elements.emplace_back(lane_element);
            }
            new_primop = world.struct_agg(newtype, rebuild_struct_elements)->as<PrimOp>();
        } else {
            new_primop = primop->rebuild(world, newtype, nops)->as<PrimOp>();
        }
    } else if (auto agg = primop->isa<Vector>()) {
        if (newtype->like(agg->type())) {
            new_primop = primop->rebuild(world, newtype, nops)->as<PrimOp>();
        } else {
            auto element_type = newtype->op(0);
            size_t vector_width = primop->type()->as<VectorType>()->length();
            //assert(element_type->isa<VariantType>());

            std::vector<const Def*> rebuild_struct_elements;
            for (size_t lane = 0; lane < vector_width; ++lane) {
                auto element = nops[lane];
                rebuild_struct_elements.emplace_back(element);
            }
            new_primop = world.struct_agg(newtype, rebuild_struct_elements)->as<PrimOp>();
        }
    } else {
        new_primop = primop->rebuild(world, newtype, nops)->as<PrimOp>();
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

    //std::cerr << "Mapping " << primop->to_string() << " to " << new_primop->to_string() << "\n";
    def2def[primop] = new_primop;

    return new_primop;
}

void Flatten::flatten_body(const Continuation *old_continuation, Continuation *new_continuation) {
    //std::cerr << "Flattening " << old_continuation->to_string() << " into " << new_continuation->to_string() << "\n";
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

const Continuation* Flatten::flatten_continuation(const Continuation* kernel) {
    //std::cerr << "Flatten Continuation " << kernel->to_string() << "\n";
    //Continuation *orig = const_cast<Continuation*>(kernel);
    //DUMP_BLOCK(orig);

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

bool Flatten::run() {
    bool unchanged = true;
    //std::cerr << "orig\n";
    //world.dump();

    //std::cerr << "flattening\n";
    for (auto continuation : world.copy_continuations()) {
        if (continuation->num_params() > 1 && continuation->param(1)->type()->isa<VectorType>() && continuation->param(1)->type()->as<VectorType>()->is_vector()) {
            if (continuation->empty() || !continuation->is_returning())
                continue;
            const Continuation* new_continuation = flatten_continuation(continuation);
            //std::cerr << "new block\n";
            Continuation *newb = const_cast<Continuation*>(new_continuation);
            //DUMP_BLOCK(newb);
            continuation->replace(newb);
            unchanged = false;
        }
    }

    //world.cleanup();

    //std::cerr << "Done flattening\n";
    //world.dump();

    return unchanged;
}

bool flatten_vectors(World& world) {
    world.VLOG("start flatten");

    bool res = Flatten(world).run();

    world.VLOG("end flatten");
    return res;
}

}
