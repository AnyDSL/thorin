#include "thorin/transform/mangle.h"
#include "thorin/transform/flatten_vectors.h"
#include "thorin/world.h"
#include "thorin/analyses/divergence.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/verify.h"

#include <llvm/Support/Timer.h>
#include <llvm/Support/raw_ostream.h>

#include <iostream>
#include <map>

//#define DUMP_WIDEN
//#define DUMP_VECTORIZER
//#define DUMP_VECTORIZER_LINEARIZER

//#define DBG_TIME(v, t) llvm::TimeRegion v(t)
#define DBG_TIME(v, t) (void)(t)

#define DUMP_BLOCK(block) block->dump(std::numeric_limits<size_t>::max())

namespace thorin {

class Vectorizer {
public:
    Vectorizer(World &world)
        : world_(world)
        , time("Vec", "Vectorize")
        , time_div("Div", "Divergence Analysis")
        , time_widen("Widen", "Widen")
        , time_clean("Cleanup", "Cleanup")
        , time_lin("Linearization", "Linearization")
        , kernel(nullptr)
        , current_scope(nullptr)
    {}
    bool run();

private:
    World& world_;

    llvm::Timer time;
    llvm::Timer time_div;
    llvm::Timer time_widen;
    llvm::Timer time_clean;
    llvm::Timer time_lin;

    ContinuationSet done_;
    Def2Def def2def_;
    DivergenceAnalysis * div_analysis_;

    size_t vector_width = 8;

    std::queue<Continuation*> queue_;
    void enqueue(Continuation* continuation) {
        if (done_.emplace(continuation).second)
            queue_.push(continuation);
    }

    const Type *widen(const Type *);
    const Def *widen(const Def*, const Continuation*);
    Continuation *widen();

    void widen_setup(Continuation *);
    Continuation *kernel;
    Scope *current_scope;
    bool widen_within(const Def *);

    void widen_body(Continuation *, Continuation *);
    Continuation* widen_head(Continuation* old_continuation);

    void linearize(Continuation * vectorized);
};

const Type *Vectorizer::widen(const Type *old_type) {
    if (old_type->isa<MemType>())
        return old_type;
    if (auto primtype = old_type->isa<PrimType>()) {
        assert(primtype->length() == 1);
        return world_.prim_type(primtype->primtype_tag(), vector_width);
    } else if (auto ptrtype = old_type->isa<PtrType>()) {
        assert(ptrtype->length() == 1);
        return world_.ptr_type(ptrtype->pointee(), vector_width);
    } else if (auto tupletype = old_type->isa<TupleType>()) {
        Array<const Type*> elements(tupletype->num_ops());
        for (size_t i = 0; i < tupletype->num_ops(); ++i) {
            auto newelement = widen(tupletype->op(i));
            elements[i] = newelement;
        }
        return world_.tuple_type(elements);
    } else {
        return world_.vec_type(old_type, vector_width);
    }
}

Continuation* Vectorizer::widen_head(Continuation* old_continuation) {
    assert(!def2def_.contains(old_continuation));
    assert(!old_continuation->empty());
    Continuation* new_continuation;

    std::vector<const Type*> param_types;

    if (div_analysis_->isPredicated[old_continuation]) {
        param_types.emplace_back(world_.mem_type());
        param_types.emplace_back(widen(world_.type_bool()));
    }

    for (size_t i = 0, e = old_continuation->num_params(); i != e; ++i) {
        if (!is_mem(old_continuation->param(i)) && div_analysis_->getUniform(old_continuation->param(i)) != DivergenceAnalysis::State::Uniform)
            param_types.emplace_back(widen(old_continuation->param(i)->type()));
        else
            if (!div_analysis_->isPredicated[old_continuation] || !is_mem(old_continuation->param(i)))
                param_types.emplace_back(old_continuation->param(i)->type());
    }

    auto fn_type = world_.fn_type(param_types);
    new_continuation = world_.continuation(fn_type, old_continuation->debug_history());

    assert(new_continuation);
    def2def_[old_continuation] = new_continuation;

    //TODO: Under some conditions, the mem parameter of other continuations might be relevant also.
    for (size_t i = 0, e = old_continuation->num_params(); i != e; ++i) {
        if (div_analysis_->isPredicated[old_continuation]) {
            if (is_mem(old_continuation->param(i))) {
                assert(is_mem(new_continuation->param(0)));
                def2def_[old_continuation->param(i)] = new_continuation->param(0);
            } else {
                int j = i;
                auto mem_param = old_continuation->mem_param();
                if (mem_param) {
                    auto mem_param_index = mem_param->index();
                    if (i < mem_param_index)
                        j += 2;
                    else
                        j += 1;
                } else {
                    j += 2;
                }
                def2def_[old_continuation->param(i)] = new_continuation->param(j);
            }
        } else {
            assert(new_continuation->param(i));
            def2def_[old_continuation->param(i)] = new_continuation->param(i);
        }
    }

    return new_continuation;
}

const Def* Vectorizer::widen(const Def* old_def, const Continuation* context) {
#ifdef DUMP_WIDEN
    std::cout << "Widen\n";
    old_def->dump();
#endif

    if (def2def_.contains(old_def)) {
#ifdef DUMP_WIDEN
        std::cout << "Found\n";
#endif
        auto new_def = def2def_[old_def];
        return new_def;
    } else if (!widen_within(old_def)) {
#ifdef DUMP_WIDEN
        std::cout << "NWithin\n";
        old_def->dump();
#endif
        if (auto cont = old_def->isa_continuation()) {
            if (cont->is_intrinsic() && cont->intrinsic() == Intrinsic::Match) {
                auto type = old_def->type();
                auto match = world_.match(widen(type->op(1)), type->num_ops() - 3);
                assert(match);
                return def2def_[old_def] = match;
            }
            if (cont->is_intrinsic() && cont->intrinsic() == Intrinsic::Branch) {
                auto mem_ty = world_.mem_type();
                auto bool_vec_ty = widen(world_.type_bool());
                auto branch = world_.continuation(world_.fn_type({mem_ty, bool_vec_ty, world_.fn_type({mem_ty, bool_vec_ty}), world_.fn_type({mem_ty, bool_vec_ty})}), Intrinsic::Branch, {"br_vec"});
                assert(branch);
                return def2def_[old_def] = branch;
            }
        }
        return old_def;
    } else if (auto old_continuation = old_def->isa_continuation()) {
#ifdef DUMP_WIDEN
        std::cout << "Contin\n";
#endif
        auto new_continuation = widen_head(old_continuation);
        widen_body(old_continuation, new_continuation);
        return new_continuation;
    } else if (div_analysis_->getUniform(old_def) == DivergenceAnalysis::State::Uniform) {
#ifdef DUMP_WIDEN
        std::cout << "Uni\n";
#endif
        //Make a copy!
        Array<const Def*> nops(old_def->num_ops());

        for (unsigned i = 0; i < old_def->num_ops(); i++) {
            nops[i] = (widen(old_def->op(i), context)); //These should all be uniform as well.
        }

        auto r = old_def->as<PrimOp>()->rebuild(world_, old_def->type(), nops);
        return r; //TODO: this def could contain a continuation inside a tuple for match cases!
    } else if (auto param = old_def->isa<Param>()) {
#ifdef DUMP_WIDEN
        std::cout << "Param\n";
#endif
        widen(param->continuation(), param->continuation());
        assert(def2def_.contains(param));
        return def2def_[param];
    } else if (auto extract = old_def->isa<Extract>()) {
#ifdef DUMP_WIDEN
        std::cout << "Extract\n";
#endif
        Array<const Def*> nops(extract->num_ops());

        nops[0] = widen(extract->op(0), context);
        if (auto vectype = nops[0]->type()->isa<VectorType>(); vectype && vectype->is_vector())
            nops[1] = world_.tuple({world_.top(extract->op(1)->type()), extract->op(1)});
        else
            nops[1] = extract->op(1);

        auto type = widen(extract->type());
        const Def* new_primop;
        if (extract->isa<PrimLit>()) {
            assert(false && "This should not be reachable!");
            Array<const Def*> elements(vector_width);
            for (size_t i = 0; i < vector_width; i++) {
                elements[i] = extract;
            }
            new_primop = world_.vector(elements, extract->debug_history());
        } else {
            new_primop = extract->as<PrimOp>()->rebuild(world_, type, nops);
        }
        assert(new_primop);
        return def2def_[extract] = new_primop;
    } else if (auto varextract = old_def->isa<VariantExtract>()) {
#ifdef DUMP_WIDEN
        std::cout << "VarExtract\n";
#endif
        Array<const Def*> nops(varextract->num_ops());

        nops[0] = widen(varextract->op(0), context);

        auto type = widen(varextract->type());
        const Def* new_primop;
        if (varextract->isa<PrimLit>()) {
            assert(false && "This should not be reachable!");
            Array<const Def*> elements(vector_width);
            for (size_t i = 0; i < vector_width; i++) {
                elements[i] = varextract;
            }
            new_primop = world_.vector(elements, varextract->debug_history());
        } else {
            new_primop = varextract->as<PrimOp>()->rebuild(world_, type, nops);
        }
        assert(new_primop);
        return def2def_[varextract] = new_primop;
    } else if (auto arithop = old_def->isa<ArithOp>()) {
#ifdef DUMP_WIDEN
        std::cout << "Arith\n";
#endif
        Array<const Def*> nops(arithop->num_ops());
        bool any_vector = false;
        for (size_t i = 0, e = arithop->num_ops(); i != e; ++i) {
            nops[i] = widen(arithop->op(i), context);
            if (auto vector = nops[i]->type()->isa<VectorType>())
                any_vector |= vector->is_vector();
            if (nops[i]->type()->isa<VariantVectorType>())
                any_vector = true;
        }

        if (any_vector) {
            for (size_t i = 0, e = arithop->num_ops(); i != e; ++i) {
                if (auto vector = nops[i]->type()->isa<VectorType>())
                    if (vector->is_vector())
                        continue;
                if (nops[i]->type()->isa<VariantVectorType>())
                    continue;
                if (nops[i]->type()->isa<MemType>())
                    continue;

                //non-vector element in a vector setting needs to be extended to a vector.
                Array<const Def*> elements(vector_width);
                for (size_t j = 0; j < vector_width; j++) {
                    elements[j] = nops[i];
                }
                nops[i] = world_.vector(elements, nops[i]->debug_history());
            }
        }

        auto type = widen(arithop->type());
        const Def* new_primop;
        if (arithop->isa<PrimLit>()) {
            Array<const Def*> elements(vector_width);
            for (size_t i = 0; i < vector_width; i++) {
                elements[i] = arithop;
            }
            new_primop = world_.vector(elements, arithop->debug_history());
        } else {
            new_primop = arithop->as<PrimOp>()->rebuild(world_, type, nops);
        }
        assert(new_primop);
        return def2def_[arithop] = new_primop;
    } else {
#ifdef DUMP_WIDEN
        std::cout << "Primop\n";
#endif
        auto old_primop = old_def->as<PrimOp>();
        Array<const Def*> nops(old_primop->num_ops());
        bool any_vector = false;
        for (size_t i = 0, e = old_primop->num_ops(); i != e; ++i) {
            nops[i] = widen(old_primop->op(i), context);
            auto oldtype = old_primop->op(i)->type();
            auto newtype_vec = widen(oldtype);
            if(newtype_vec == oldtype)
                continue;
            auto newtype_actual = nops[i]->type();
            if (newtype_vec == newtype_actual)
                any_vector = true;
            if (newtype_actual != oldtype)
                any_vector = true;

            if (auto vectype = nops[i]->type()->isa<VectorType>(); vectype && vectype->is_vector())
                assert(any_vector);
                //any_vector = true;
            if (nops[i]->type()->isa<VariantVectorType>())
                assert(any_vector);
                //any_vector = true;
        }

        if (any_vector && (old_primop->isa<BinOp>() || old_primop->isa<Select>() || old_primop->isa<StructAgg>() || old_primop->isa<Access>())) {
            for (size_t i = 0, e = old_primop->num_ops(); i != e; ++i) {
                auto oldtype = old_primop->op(i)->type();
                auto newtype_vec = widen(oldtype);
                if(newtype_vec == oldtype)
                    continue;
                auto newtype_actual = nops[i]->type();
                if (newtype_vec == newtype_actual)
                    continue;

                if (auto vectype = nops[i]->type()->isa<VectorType>(); vectype && vectype->is_vector())
                    THORIN_UNREACHABLE;
                    //continue;
                if (nops[i]->type()->isa<VariantVectorType>())
                    THORIN_UNREACHABLE;
                    //continue;
                if (nops[i]->type()->isa<MemType>())
                    continue;

                Array<const Def*> elements(vector_width);
                for (size_t j = 0; j < vector_width; j++)
                    elements[j] = nops[i];
                nops[i] = world_.vector(elements, nops[i]->debug_history());
            }
        }

        if (old_def->isa<Slot>()) {
            //force creation of a vectorized slot
            any_vector = true;
        }

        const Type* type;
        if (any_vector)
          type = widen(old_primop->type());
        else
          type = old_primop->type();

        const Def* new_primop;

        if (old_primop->isa<PrimLit>()) {
            assert(false && "Primlits are uniform");

        } else if (old_primop->isa<Access>()) {
            if (div_analysis_->isPredicated[const_cast<Continuation*>(context)]) {
                auto new_cont = def2def_[context]->isa_continuation();
                assert(new_cont);

                new_cont->param(1);

                if (old_primop->isa<Store>()) {
                    new_primop = world_.maskedstore(nops[0], nops[1], nops[2], new_cont->param(1));
                } else if (old_primop->isa<Load>()) {
                    new_primop = world_.maskedload(nops[0], nops[1], new_cont->param(1));
                } else {
                    THORIN_UNREACHABLE;
                }
            } else {
                new_primop = old_primop->rebuild(world_, type, nops);
            }
        } else {
            new_primop = old_primop->rebuild(world_, type, nops);
        }
        if (old_def->isa<Slot>()) {
            assert(new_primop->type() == type);
            auto vectype = new_primop->type()->isa<VectorType>(); 
            assert(vectype && vectype->is_vector());
        }
        assert(new_primop);
        if (!new_primop->type()->like(type)) {
            old_primop->dump();
            new_primop->type()->dump();
            old_primop->type()->dump();
            type->dump();
            std::cerr << any_vector << "\n";
        }
        assert(new_primop->type()->like(type));
        return def2def_[old_primop] = new_primop;
    }
}

void Vectorizer::widen_body(Continuation* old_continuation, Continuation* new_continuation) {
    assert(!old_continuation->empty());
#ifdef DUMP_WIDEN
    std::cout << "Body\n";
#endif

    // fold branch and match
    if (auto callee = old_continuation->callee()->isa_continuation()) {
        switch (callee->intrinsic()) {
            case Intrinsic::Branch: {
                if (auto lit = widen(old_continuation->arg(0), old_continuation)->isa<PrimLit>()) {
                    auto cont = lit->value().get_bool() ? old_continuation->arg(1) : old_continuation->arg(2);
                    return new_continuation->jump(widen(cont, old_continuation), {}, old_continuation->debug());
                }
                break;
            }
            case Intrinsic::Match:
                if (old_continuation->num_args() == 2)
                    return new_continuation->jump(widen(old_continuation->arg(1), old_continuation), {}, old_continuation->debug());

                if (auto lit = widen(old_continuation->arg(0), old_continuation)->isa<PrimLit>()) {
                    for (size_t i = 2; i < old_continuation->num_args(); i++) {
                        auto new_arg = widen(old_continuation->arg(i), old_continuation);
                        if (world_.extract(new_arg, 0_s)->as<PrimLit>() == lit)
                            return new_continuation->jump(world_.extract(new_arg, 1), {}, old_continuation->debug());
                    }
                }
                break;
            default:
                break;
        }
    }

    Array<const Def*> nops(old_continuation->num_ops());
    for (size_t i = 0, e = nops.size(); i != e; ++i)
        nops[i] = widen(old_continuation->op(i), old_continuation);

    const Def* ntarget = nops.front();   // new target of new_continuation

    if (auto callee = old_continuation->callee()->isa_continuation()) {
        if (callee->is_imported()) {
            auto old_fn_type = callee->type()->as<FnType>();
            Array<const Type*> ops(old_fn_type->num_ops());
            bool any_vector = false;
            size_t vector_width = 1;
            for (size_t i = 0; i < old_fn_type->num_ops(); i++) {
                ops[i] = nops[i + 1]->type();
                if (auto vectype = ops[i]->isa<VectorType>(); vectype && vectype->is_vector()) {
                    any_vector = true;
                    assert((vector_width == 1 || vector_width == vectype->length()) && "No mixed vector types allowed.");
                    vector_width = vectype->length();
                }
            }
            Debug de = callee->debug();

            std::stringstream vi1_suffix;
            vi1_suffix << "v" << vector_width << "i1";
            std::stringstream vp0i8_suffix;
            vp0i8_suffix << "v" << vector_width << "p0i8";
            std::stringstream vf32_suffix;
            vf32_suffix << "v" << vector_width << "f32";
            std::stringstream vf64_suffix;
            vf64_suffix << "v" << vector_width << "f64";

            if (de.name.rfind("rv_", 0) == 0) {
                //leave unchanged, will be lowered in backend.
                //std::cerr << "RV intrinsic for BE: " << de.name << "\n";
                ntarget = world_.continuation(world_.fn_type(ops), Continuation::Attributes(Intrinsic::RV), de);
                //ntarget = world_.continuation(world_.fn_type(ops), callee->attributes(), de);
            } else if (any_vector) {
                if (de.name == "llvm.exp.f32")
                    de.name = "llvm.exp." + vf32_suffix.str();
                else if (de.name == "llvm.exp.f64")
                    de.name = "llvm.exp." + vf64_suffix.str();
                else if (de.name == "llvm.sqrt.f32")
                    de.name = "llvm.sqrt." + vf32_suffix.str();
                else if (de.name == "llvm.sqrt.f64")
                    de.name = "llvm.sqrt." + vf64_suffix.str();
                else if (de.name == "llvm.sin.f32")
                    de.name = "llvm.sin." + vf32_suffix.str();
                else if (de.name == "llvm.sin.f64")
                    de.name = "llvm.sin." + vf64_suffix.str();
                else if (de.name == "llvm.cos.f32")
                    de.name = "llvm.cos." + vf32_suffix.str();
                else if (de.name == "llvm.cos.f64")
                    de.name = "llvm.cos." + vf64_suffix.str();
                else if (de.name == "llvm.minnum.f32")
                    de.name = "llvm.minnum." + vf32_suffix.str();
                else if (de.name == "llvm.minnum.f64")
                    de.name = "llvm.minnum." + vf64_suffix.str();
                else if (de.name == "llvm.floor.f32")
                    de.name = "llvm.floor." + vf32_suffix.str();
                else if (de.name == "llvm.floor.f64")
                    de.name = "llvm.floor." + vf64_suffix.str();
                else if (de.name == "llvm.expect.i1")
                    de.name = "llvm.expect." + vi1_suffix.str();
                else if (de.name == "llvm.prefetch.p0i8")
                    de.name = "llvm.prefetch." + vp0i8_suffix.str();
                else if (de.name == "llvm.fabs.f32")
                    de.name = "llvm.fabs." + vf32_suffix.str();
                else {
                    std::cerr << "Not supported: " << de.name << "\n";
                    assert(false && "Import not supported in vectorize.");
                }

                for (size_t i = 0, e = old_fn_type->num_ops(); i != e; ++i) {
                    if (auto vector = ops[i]->isa<VectorType>())
                        if (vector->is_vector())
                            continue;
                    if (ops[i]->isa<VariantVectorType>())
                        continue;
                    if (ops[i]->isa<MemType>())
                        continue;
                    if (ops[i]->isa<FnType>())
                        continue;

                    //non-vector element in a vector setting needs to be extended to a vector.
                    Array<const Def*> elements(vector_width);
                    for (size_t j = 0; j < vector_width; j++) {
                        elements[j] = nops[i + 1];
                    }
                    nops[i + 1] = world_.vector(elements, nops[i + 1]->debug_history());
                    ops[i] = nops[i + 1]->type();
                }

                ntarget = world_.continuation(world_.fn_type(ops), callee->attributes(), de);
            } else {
                ntarget = world_.continuation(world_.fn_type(ops), callee->attributes(), de);
            }
            assert(ntarget);
            def2def_[callee] = ntarget;
        } else {
            if(ntarget == old_continuation->callee()) {

                bool predicated = div_analysis_->isPredicated[old_continuation];
                const Def* pred_param;
                if (predicated) {
                    pred_param = new_continuation->param(1);
                }

                auto cont = ntarget->isa_continuation();
                assert(cont);

                auto return_param = cont->ret_param();
                assert(return_param);
                auto new_return_param = nops[return_param->index() + 1];
                assert(new_return_param);

                auto num_parameters = cont->type()->num_ops();
                auto narg_types = std::vector<const Type*>(num_parameters);
                for (size_t i = 0; i < num_parameters; i++) {
                    narg_types[i] = nops[i + 1]->type();
                }
                auto cascade = world_.continuation(world_.fn_type(narg_types), {"cascade"});
                size_t vector_width = 8;

                Array <const Def*> args (num_parameters);
                auto current_lane = cascade;

                Array<const Def*> return_cache(vector_width);

                for (size_t i = 0; i < vector_width; i++) {
                    auto parameters = std::vector<Def*>(num_parameters);
                    for (size_t j = 0; j < num_parameters; j++) {
                        const Def* arg = cascade->param(j);
                        assert(arg);
                        if (arg->type()->isa<VectorType>() && arg->type()->as<VectorType>()->is_vector()) {
                            arg = world_.extract(arg, world_.literal_qs32(i, {}));
                        }
                        args[j] = arg;
                    }

                    auto next_lane = world_.continuation(cont->param(6)->type()->as<FnType>(), {"cascade"});

                    auto return_parameter = 6;
                    args[return_parameter] = next_lane;

                    if (predicated) {
                        auto mask = world_.extract(pred_param, world_.literal_qs32(i, {}));

                        auto then_cont = world_.continuation(world_.fn_type({world_.mem_type()}), {"then"});
                        auto next_cont = world_.continuation(world_.fn_type({world_.mem_type()}), {"next"});
                        auto return_cont = world_.continuation(args[num_parameters - 1]->type()->as<FnType>(), {"in_return"});

                        current_lane->branch(args[0], mask, then_cont, next_cont);

                        Array <const Def*> inner_args (num_parameters);
                        inner_args[0] = then_cont->param(0);
                        for (size_t n = 1; n < num_parameters - 1; n++)
                            inner_args[n] = args[n];
                        inner_args[num_parameters - 1] = return_cont;

                        then_cont->jump(cont, inner_args);

                        next_cont->jump(return_cont, {next_cont->param(0), world_.literal_qs32(0, {})});

                        current_lane = return_cont;
                    } else {
                        current_lane->jump(cont, args);
                        current_lane = next_lane;
                    }

                    return_cache[i] = current_lane->param(1);
                }

                const Def* return_value;
                if (new_return_param->type()->op(2)->isa<VectorType>()) {
                    return_value = world_.vector(return_cache, {});
                } else {
                    return_value = return_cache[vector_width - 1];
                }

                const Def* out_predicate;

                if (div_analysis_->isPredicated[old_continuation]) {
                    out_predicate = pred_param;
                } else {
                    auto true_elem = world_.literal_bool(true, {});
                    Array<const Def *> elements(vector_width);
                    for (size_t i = 0; i < vector_width; i++)
                        elements[i] = true_elem;
                    out_predicate = world_.vector(elements);
                }

                current_lane->jump(new_return_param, {current_lane->param(0), out_predicate, return_value});

                ntarget = cascade;
            }
        }
    }

    Scope scope(kernel);

#ifdef DUMP_WIDEN
    std::cerr << "Ntarget:\n";
    ntarget->dump();
    ntarget->type()->dump();
#endif
    Array <const Def*> nargs (ntarget->type()->num_ops());

    if (old_continuation->op(0)->isa<Continuation>() && scope.contains(old_continuation->op(0))) {
        Continuation* oldtarget = const_cast<Continuation*>(old_continuation->op(0)->as<Continuation>());

        for (size_t i = 0; i < nops.size() - 1; i++) {
            auto arg = nops[i + 1];
            int  j = i;

            if (div_analysis_->isPredicated[oldtarget]) {
                if (i > 0)
                    j += 1;
            }

            if (!is_mem(arg) &&
                    (!arg->type()->isa<VectorType>() ||
                     !arg->type()->as<VectorType>()->is_vector()) && //TODO: This is not correct.
                    div_analysis_->getUniform(oldtarget->param(i)) != DivergenceAnalysis::State::Uniform) {
                Array<const Def*> elements(vector_width);
                for (size_t i = 0; i < vector_width; i++) {
                    elements[i] = arg;
                }

                auto new_param = world_.vector(elements, arg->debug_history());
                nargs[j] = new_param;
            } else {
                nargs[j] = arg;
            }
        }

        if (div_analysis_->isPredicated[oldtarget]) {
            //TODO: The memory parameter might also be subject to further investigation here.
            auto param = new_continuation->param(1);
            if(!div_analysis_->isPredicated[old_continuation]) {
                assert (!(param->type()->isa<PrimType>() && param->type()->as<PrimType>()->is_vector() && param->type()->as<PrimType>()->primtype_tag() == PrimTypeTag::PrimType_bool));
                auto true_elem = world_.literal_bool(true, {});
                Array<const Def *> elements(vector_width);
                for (size_t i = 0; i < vector_width; i++)
                    elements[i] = true_elem;
                auto one_predicate = world_.vector(elements);
                nargs[1] = one_predicate;
            } else {
                assert(param->type()->isa<PrimType>() && param->type()->as<PrimType>()->is_vector() && param->type()->as<PrimType>()->primtype_tag() == PrimTypeTag::PrimType_bool);
                nargs[1] = new_continuation->param(1);
            }
        }
    } else {
        for (size_t i = 0; i < nops.size() - 1; i++)
            nargs[i] = nops[i + 1];
    }

    for (size_t i = 0, e = nops.size() - 1; i != e; ++i) {
            //if (div_analysis_->getUniform(old_continuation->op(i)) == DivergenceAnalysis::State::Uniform &&
            //        div_analysis_->getUniform(oldtarget->param(i-1)) != DivergenceAnalysis::State::Uniform) {
            if ((!nargs[i]->type()->isa<VectorType>() || !nargs[i]->type()->as<VectorType>()->is_vector()) &&
                    ntarget->isa<Continuation>() &&
                    ntarget->as<Continuation>()->param(i)->type()->isa<VectorType>() &&
                    ntarget->as<Continuation>()->param(i)->type()->as<VectorType>()->is_vector()) { //TODO: base this on divergence analysis
                Array<const Def*> elements(vector_width);
                for (size_t j = 0; j < vector_width; j++)
                    elements[j] = nargs[i];
                nargs[i] = world_.vector(elements, nargs[i]->debug_history());
            }
    }

    if (ntarget->isa<Continuation>() && ntarget->as<Continuation>()->is_intrinsic() && (ntarget->as<Continuation>()->intrinsic() == Intrinsic::Match || ntarget->as<Continuation>()->intrinsic() == Intrinsic::Branch)) {
        for (size_t i = 0, e = nops.size() - 1; i != e; ++i) {
            auto ntarget_type = ntarget->type()->op(i);
            auto nargs_type = nargs[i]->type();
            if (ntarget_type != nargs_type) {
                assert(i <= 4 && "Only seen with 'otherwise' for match, or true/false for branch");
                assert(ntarget_type->isa<FnType>());
                assert(nargs_type->isa<FnType>());
                assert(ntarget_type->num_ops() == 2);
                assert(nargs_type->num_ops() == 1);

                auto bool_vec_ty = widen(world_.type_bool());
                auto buffer_continuation = world_.continuation(world_.fn_type({world_.mem_type(), bool_vec_ty}));

                buffer_continuation->jump(nargs[i], {buffer_continuation->param(0)});

                nargs[i] = buffer_continuation;
            }
        }
    }

    new_continuation->jump(ntarget, nargs, old_continuation->debug());
#ifdef DUMP_WIDEN
    std::cout << "Jump\n";
#endif
}

Continuation *Vectorizer::widen() {
    Continuation *ncontinuation;

    // create new_entry - but first collect and specialize all param types
    std::vector<const Type*> param_types;
    for (size_t i = 0, e = kernel->num_params(); i != e; ++i) {
        if (i == 1) {
            param_types.emplace_back(widen(kernel->param(i)->type()));
        } else {
            param_types.emplace_back(kernel->param(i)->type());
        }
    }

    auto fn_type = world_.fn_type(param_types);
    ncontinuation = world_.continuation(fn_type, kernel->debug_history());

    // map value params
    assert(kernel);
    def2def_[kernel] = kernel;
    for (size_t i = 0, j = 0, e = kernel->num_params(); i != e; ++i) {
        auto old_param = kernel->param(i);
        auto new_param = ncontinuation->param(j++);
        assert(new_param);
        def2def_[old_param] = new_param;
        new_param->debug().name = old_param->name();
    }

    // mangle filter
    if (!kernel->filter().empty()) {
        Array<const Def*> new_filter(ncontinuation->num_params());
        size_t j = 0;
        for (size_t i = 0, e = kernel->num_params(); i != e; ++i) {
            new_filter[j++] = widen(kernel->filter(i), kernel);
        }

        for (size_t e = ncontinuation->num_params(); j != e; ++j)
            new_filter[j] = world_.literal_bool(false, Debug{});

        ncontinuation->set_filter(new_filter);
    }

    widen_body(kernel, ncontinuation);

    return ncontinuation;
}

void Vectorizer::widen_setup(Continuation* kern) {
    kernel = kern;
    if (current_scope)
        delete current_scope;
    current_scope = new Scope(kernel);
}

bool Vectorizer::widen_within(const Def* def) {
    return current_scope->contains(def);
}

class MemCleaner {
public:
    MemCleaner(const Continuation* cont) {
        for (auto& block : schedule(Scope(const_cast<Continuation*>(cont)))) {
            conts.push(block);
        }
    }

    void run();
    void run(const Def*);

    unique_queue<ContinuationSet> conts;
    DefSet defs;
};

void MemCleaner::run(const Def* def) {
    if (def->no_dep() || !defs.emplace(def).second) return;

    const Def* replacement = nullptr;
    for (auto use : def->uses()) {
        if (auto extract = use->isa<Extract>(); extract && extract->op(0) == def) {
            if (extract->index()->isa<PrimLit>()) {
                auto index = extract->index()->as<PrimLit>()->qu32_value();
                if (index == ((qu32)0)) {
                    if (replacement) {
                        //std::cerr << "Replacing " << extract->unique_name() << " with " << replacement->unique_name() << "\n";
                        extract->replace(replacement);
                    } else {
                        replacement = extract;
                    }
                }
            }
        }
    }

    for (auto op : def->ops()) {
        if (auto cont = op->isa_continuation()) {
            conts.push(cont);
        } else {
            run(op);
        }
    }
}

void MemCleaner::run() {
    while (!conts.empty()) {
        auto cont = conts.pop();

        if (!cont->empty()) {
            run(cont);
        }
    }
}

void Vectorizer::linearize(Continuation * vectorized) {
#ifdef DUMP_VECTORIZER_LINEARIZER
    std::cerr << "\nPre lin\n";
    DUMP_BLOCK(vectorized);
    std::cerr << "\n";
#endif

    std::queue <Continuation*> split_queue;
    GIDMap<Continuation*, ContinuationSet> encloses_splits;
    GIDMap<const Continuation*, const Def*> runningVars;

    //TODO: find enter definition if already present.
    const Enter* enter = nullptr;
    const Def* current_frame = nullptr;

    { // Gather split nodes
        ContinuationSet done;

        //TODO: Do not(!) use relJoins for linearization. Check all Branches and Loops for divergence instead!
        //TODO: Write a more general approach to predicated execution.
        for (auto it : div_analysis_->relJoins) {
            if (div_analysis_->splitParrents.contains(it.first) && div_analysis_->splitParrents[it.first].size()) {
                for (auto encloses : div_analysis_->splitParrents[it.first]) {
                    encloses_splits[encloses].emplace(it.first);
                }
            } else {
#ifdef DUMP_VECTORIZER_LINEARIZER
                std::cerr << "Found split node: ";
                it.first->dump();
#endif
                auto cont = it.first->op(0)->isa_continuation();
                if (cont && cont->is_intrinsic() && (cont->intrinsic() == Intrinsic::Match || cont->intrinsic() == Intrinsic::Branch)) {
                    auto new_cont = def2def_[it.first]->as<Continuation>();
                    assert(new_cont);
                    if (new_cont->arg(1)->type()->isa<VectorType>() && new_cont->arg(1)->type()->as<VectorType>()->is_vector()) {
                        split_queue.push(it.first);
                        done.emplace(it.first);
                    }
                }
            }
        }

        for (auto it : div_analysis_->loopBodies) {
            if (!current_frame) {
                auto mem = vectorized->mem_param();
                if (!enter)
                    enter = world_.enter(mem)->as<Enter>();
                current_frame = enter->out_frame();
                auto newmem = enter->out_mem();
                for (auto use : mem->copy_uses()) {
                    auto use_inst = use.def();
                    if (use_inst == enter)
                        continue;
                    assert(use_inst);
                    int index = use.index();
                    Def* olduse = const_cast<Def*>(use_inst);
                    olduse->unset_op(index);
                    olduse->set_op(index, newmem);
                }
            }

            assert(current_frame);
            auto bool_vec_ty = world_.type_bool(vector_width);
            auto running_var = world_.slot(bool_vec_ty, current_frame, Debug("loop_running"));
            assert(running_var);
            runningVars[def2def_[it.first]->as<Continuation>()] = running_var;

            if (done.emplace(it.first).second) {
#ifdef DUMP_VECTORIZER_LINEARIZER
                std::cerr << "Adding missing block";
                it.first->dump();
#endif
                split_queue.push(it.first);
            }
        }
    }

    while (!split_queue.empty()) {
        Continuation* latch_old = pop(split_queue);
        ContinuationSet joins_old = div_analysis_->relJoins[latch_old];
        assert (!div_analysis_->splitParrents.contains(latch_old));
        Continuation * latch = const_cast<Continuation*>(def2def_[latch_old]->as<Continuation>());
        assert(latch);

        auto cont = latch->op(0)->isa_continuation();

#ifdef DUMP_VECTORIZER_LINEARIZER
        std::cerr << "Linearize Node\n";
        DUMP_BLOCK(latch_old);
        for (auto join : joins_old)
            join->dump();
        DUMP_BLOCK(latch);
#endif

        bool isLoopHeader = false;
        bool isLoopExit = false;
        if (cont && div_analysis_->loopBodies.lookup(latch_old).has_value()) {
            isLoopHeader = true;
        }
        if (isLoopHeader) {
            auto exits = div_analysis_->loopExits[latch_old];
            for (auto succ : latch_old->succs()) {
                if (exits.find(succ) != exits.end()) {
                    isLoopExit = true;
                }
            }
        } else {
            for (auto it : div_analysis_->loopExits) {
                for (auto succ : latch_old->succs()) {
                    if (it.second.find(succ) != it.second.end()) {
                        isLoopExit = true;
                    }
                }
            }
        }

#ifdef DUMP_VECTORIZER_LINEARIZER
        DUMP_BLOCK(vectorized);
#endif

        if (isLoopHeader) {
            if(isLoopExit)
                assert(cont->intrinsic() == Intrinsic::Branch);

#ifdef DUMP_VECTORIZER_LINEARIZER
            std::cerr << "Reached loop header\n";
            latch_old->dump();
            latch->dump();
            DUMP_BLOCK(vectorized);
#endif

            auto exits = div_analysis_->loopExits[latch_old];
#ifdef DUMP_VECTORIZER_LINEARIZER
            for (auto exit_old : exits) {
                auto exit_new = def2def_[exit_old];
                std::cerr << exit_old->to_string() << " " << exit_new->to_string() << "\n";
            }
#endif

            for (auto exit_old : exits) {
                auto exit_new = def2def_[exit_old];

                auto exit_new_continuation = exit_new->isa_continuation();
                assert(exit_new_continuation);

                int contained_predecessors = 0;
                for (auto pred_old : exit_old->isa_continuation()->preds()) {
                    if (div_analysis_->loopBodies[latch_old].contains(const_cast<Continuation*>(pred_old)))
                        contained_predecessors++;
                }
                if(contained_predecessors != 1) {
                    std::cerr << "\e[31mError during lin\e[39m\n";
                    DUMP_BLOCK(latch_old);
                    std::cerr << "\e[35mnew Latch\e[39m\n";
                    DUMP_BLOCK(latch);
                    std::cerr << "\e[35mend\e[39m\n";
                    exit_old->dump();
                    exit_new->dump();
                    std::cerr << "contains " << contained_predecessors << " predecessors\n";
                }
                //assert(contained_predecessors == 1);
            }

            const Def * running_var;
            running_var = runningVars[latch];
            assert(running_var);

            //Set loop_running to true for all predecessors of loop head that are not inside the loop.
            for (auto pred_old : latch_old->preds()) {
                if (div_analysis_->loopBodies[latch_old].contains(pred_old))
                    continue;
                auto pred = const_cast<Continuation*>(def2def_[pred_old]->as<Continuation>());

                assert(pred->arg(0)->type()->isa<MemType>());
                const Def * mem = pred->arg(0);
                assert(mem);
                assert(is_mem(mem));

                if(div_analysis_->isPredicated[pred_old]) {
                    auto new_running_pred = pred->param(1);
                    mem = world_.store(mem, running_var, new_running_pred);
                } else {
                    auto true_elem = world_.literal_bool(true, {});
                    Array<const Def *> elements(vector_width);
                    for (size_t i = 0; i < vector_width; i++)
                        elements[i] = true_elem;
                    auto true_predicate = world_.vector(elements);

                    mem = world_.store(mem, running_var, true_predicate);
                }
                pred->unset_op(1);
                pred->set_op(1, mem);
            }

            //Exiting branches will change the "running" variable instead of leaving imediately.
            //Use predicated to work with this
            //
            //Warning: This has some strange effects on how predicates are implemented in the backend.
            //Most notably, we need to build a Phi-Node to support this.
            auto vec_mask = widen(world_.type_bool());
#ifdef DUMP_VECTORIZER_LINEARIZER
            std::cerr << "Loop exits\n";
#endif
            for (auto exit_old : exits) {
                bool smallerloopsfound = false;
                bool largerloopsfound = false;
                std::vector<Continuation*> largerloops;
                largerloops.emplace_back(latch_old);
                //TODO: After a loop exits, we need to reload the predicate of a larger loop!
                //TODO: Back-Edges that set predicates are not handled correctly in backend. This can be done by ensuring the loop predicate to be handled by a phi node if multiple predicated entries are present.
                //If such a predicate is present: ensure that all jumps to the predicated block set a propper predicate, probably use the currently present predicate if no other predicate is set manually.
                auto loopBody = div_analysis_->loopBodies[latch_old];
                for (auto loop : div_analysis_->loopExits) {
                    if (loop.first == latch_old)
                        continue;
                    if (loop.second.contains(exit_old)) {
                        //figure out if larger or smaller loop
                        auto other_loopBody = div_analysis_->loopBodies[loop.first];
                        bool larger = other_loopBody.contains(latch_old);
                        bool smaller = loopBody.contains(loop.first);

                        if (smaller) {
                            smallerloopsfound = true;
                        }
                        if (larger) {
                            largerloopsfound = true;
                            largerloops.emplace_back(loop.first);
                        }
                    }
                }

                if (smallerloopsfound) {
                    continue;
                }
                if (largerloopsfound) {
                    // build a chain of loops
                    sort(largerloops.begin(), largerloops.end(), [&](Continuation * a, Continuation * b) {
                        return a != b && div_analysis_->loopBodies[a].contains(b);
                    });

#ifdef DUMP_VECTORIZER_LINEARIZER
                    latch_old->dump();
                    std::cerr << "Found larger loops\n";
                    for (auto loop : largerloops)
                        loop->dump();
                    std::cerr << "\n";
#endif
                }

                std::vector<const Continuation*> outer_loops;
                for (auto loop : div_analysis_->loopExits) {
                    if (loop.first == latch_old)
                        continue;
                    auto loopBody = div_analysis_->loopBodies[loop.first];
                    bool larger = loopBody.contains(latch_old);
                    bool share_exit = loop.second.contains(exit_old);
                    if (larger && !share_exit) {
                        outer_loops.emplace_back(loop.first);
                    }
                }

                for (auto pred_old : exit_old->preds()) {
                    //if smaller loops are present:
                    //  put everything in a dedicated return-block.
                    if (!div_analysis_->loopBodies[latch_old].contains(const_cast<Continuation*>(pred_old)))
                        continue;
                    auto pred = const_cast<Continuation*>(def2def_[pred_old]->as<Continuation>());
                    auto index_pred = is_mem(pred->op(1)) ? 2 : 1;
                    auto index_then = index_pred + 1;
                    auto index_else = index_then + 1;

                    const Def* predicate = pred->op(index_pred);

                    if ((!predicate->type()->isa<VectorType>()) || (!predicate->type()->as<VectorType>()->is_vector())) {
                        //TODO: This is an indication of a uniform predicate!
                        Array<const Def*> elements(vector_width);
                        for (size_t i = 0; i < vector_width; i++)
                            elements[i] = predicate;
                        predicate = world_.vector(elements);
                    }

                    Continuation * loop_continue_old = const_cast<Continuation*>(pred_old->op(index_then)->as<Continuation>());
                    Continuation * loop_exit_old;
                    if (!div_analysis_->loopBodies[latch_old].contains(loop_continue_old)) {
                        predicate = world_.arithop_not(predicate);
                        loop_exit_old = loop_continue_old;
                        loop_continue_old = const_cast<Continuation*>(pred_old->op(index_else)->as<Continuation>());
                    } else {
                        loop_exit_old = const_cast<Continuation*>(pred_old->op(index_else)->as<Continuation>());
                    }

                    assert (div_analysis_->loopBodies[latch_old].contains(loop_continue_old));
                    assert (!div_analysis_->loopBodies[latch_old].contains(loop_exit_old));

                    Continuation * loop_continue = const_cast<Continuation*>(def2def_[loop_continue_old]->as<Continuation>());
                    Continuation * loop_exit = const_cast<Continuation*>(def2def_[loop_exit_old]->as<Continuation>());

                    assert(pred->arg(0)->type()->isa<MemType>());
                    const Def * mem = pred->arg(0);
                    assert(mem);
                    assert(is_mem(mem));

                    const Def* current_running_inner = nullptr;

                    for (auto loop_old : largerloops) {
                        Continuation *loop = const_cast<Continuation*>(def2def_[loop_old]->as<Continuation>());
                        auto running_var = runningVars[loop];
                        assert(running_var);

                        auto load = world_.load(mem, running_var);
                        mem = world_.extract(load, (int)0);
                        auto current_running = world_.extract(load, 1);

                        current_running = world_.arithop_and(current_running, predicate);

                        mem = world_.store(mem, running_var, current_running);

                        if (loop_old == latch_old)
                            current_running_inner = current_running;
                    }

                    assert(current_running_inner);

                    auto new_jump = world_.predicated(vec_mask);

                    pred->jump(new_jump, { mem, current_running_inner, loop_continue, loop_exit } );
                }
            }

#ifdef DUMP_VECTORIZER_LINEARIZER
            DUMP_BLOCK(vectorized);
#endif
            continue;
        } else if (isLoopExit) {
            assert(cont->intrinsic() == Intrinsic::Branch || cont->intrinsic() == Intrinsic::Predicated);
            continue;
        }

        //cases according to latch: match and branch.
        //match:
        if (cont && cont->is_intrinsic() && cont->intrinsic() == Intrinsic::Match && latch->arg(1)->type()->isa<VectorType>() && latch->arg(1)->type()->as<VectorType>()->is_vector()) {

            //TODO: This should not be needed.
            Continuation *join;
            {
                assert (joins_old.size() > 0);
                if (joins_old.size() == 1) {
                    auto join_old = *joins_old.begin();
                    join = const_cast<Continuation*>(def2def_[join_old]->as<Continuation>());
                    assert(join);
                } else {
                    //TODO: This case is probably impossible with a match node.
                    Continuation* join_old = nullptr;
                    for (Continuation* join_it : joins_old) {
#ifdef DUMP_VECTORIZER_LINEARIZER
                        join_it->dump();
#endif
                        if (!join_old || div_analysis_->sinkedBy[join_old].contains(join_it)) {
                            join_old = join_it;
                        }
                    }
                    assert(join_old);
                    join = const_cast<Continuation*>(def2def_[join_old]->as<Continuation>());
                    assert(join);
                }
            }

            assert(is_mem(latch->arg(0)));

            assert(joins_old.size() <= 1 && "no complex controll flow match");

#ifdef DUMP_VECTORIZER_LINEARIZER
            DUMP_BLOCK(vectorized);
#endif

            auto vec_mask = widen(world_.type_bool());
            auto variant_index = latch->arg(1);

            //Allocate cache for overwritten objects.
            size_t cache_size = join->num_params();
            bool join_is_first_mem = false;
            if (cache_size && is_mem(join->param(0))) {
                cache_size -= 1;
                join_is_first_mem = true;
            }

            assert(join_is_first_mem);

            Array<const Def *> join_cache(cache_size);
            {
                assert(latch->arg(0)->type()->isa<MemType>());
                const Def* mem = latch->arg(0);
                assert(mem);
                assert(is_mem(mem));

                if (cache_size && !current_frame) {
                    auto mem = vectorized->mem_param();
                    if (!enter)
                        enter = world_.enter(mem)->as<Enter>();
                    current_frame = enter->out_frame();
                    auto newmem = enter->out_mem();
                    for (auto use : mem->copy_uses()) {
                        auto use_inst = use.def();
                        if (use_inst == enter)
                            continue;
                        assert(use_inst);
                        int index = use.index();
                        Def* olduse = const_cast<Def*>(use_inst);
                        olduse->unset_op(index);
                        olduse->set_op(index, newmem);
                    }
                }

                for (size_t i = 0; i < cache_size; i++) {
                    assert(current_frame);
                    auto t = world_.slot(join->param(i + 1)->type(), current_frame, Debug("join_cache_match"));
                    join_cache[i] = t;
                }

                latch->update_arg(0, mem);
            }

            //Larger cases might be possible?
            assert(encloses_splits[latch_old].size() <= 1);

            Array<const Continuation*> splits(encloses_splits[latch_old].size()+1);
            splits[0] = latch_old;
            size_t i = 1;
            for (auto split_old : encloses_splits[latch_old])
                splits[i++] = split_old;
            std::sort (splits.begin(), splits.end(), [&](const Continuation *ac, const Continuation *bc) {
                    Continuation *a = const_cast<Continuation*>(ac);
                    Continuation *b = const_cast<Continuation*>(bc);
                    return div_analysis_->dominatedBy[a].contains(b);
                });

            for (size_t current_split_index = 0; current_split_index < splits.size(); current_split_index++) {
                auto split_old = splits[current_split_index];
                auto split = const_cast<Continuation*>(def2def_[split_old]->as<Continuation>());
                assert(split);

                Array<const Def*> local_predicates(split->num_args() - 2);
                for (size_t i = 1; i < split->num_args() - 2; i++) {
                    auto elem = split->arg(i + 2);
                    auto val = elem->as<Tuple>()->op(0)->as<PrimLit>();
                    Array<const Def *> elements(vector_width);
                    for (size_t i = 0; i < vector_width; i++) {
                        elements[i] = val;
                    }
                    auto val_vec = world_.vector(elements);
                    auto pred = world_.cmp(Cmp_eq, variant_index, val_vec);
                    local_predicates[i] = pred;
                }
                local_predicates[0] = local_predicates[1];
                for (size_t i = 2; i < split->num_args() - 2; i++) {
                    local_predicates[0] = world_.binop(ArithOp_or, local_predicates[0], local_predicates[i]);
                }
                local_predicates[0] = world_.arithop_not(local_predicates[0]);

                Array<const Def*> split_predicates = local_predicates;

                Continuation * otherwise = const_cast<Continuation*>(split->arg(2)->as<Continuation>());
                assert(otherwise);

                Continuation* join_old = *joins_old.begin();
                const Continuation* join = def2def_[join_old]->as<Continuation>();
                assert(join);

                Continuation * current_case = otherwise;

                const Def * otherwise_old = split_old->arg(2);
                Continuation * case_old = const_cast<Continuation*>(otherwise_old->as<Continuation>());

                auto bool_vec_ty = widen(world_.type_bool());
                Continuation * case_back = world_.continuation(world_.fn_type({world_.mem_type(), bool_vec_ty}), Debug("otherwise_back"));
            
                auto new_jump_split = world_.predicated(vec_mask);
                assert(current_case != case_back);
                { //mem scope
                    assert(!split_predicates[0]->isa<Vector>());
                    const Def * mem = split->arg(0);
                    assert(is_mem(mem));
                    split->jump(new_jump_split, { mem, split_predicates[0], current_case, case_back }, split->debug());
                }

                Continuation * pre_join = world_.continuation(world_.fn_type({world_.mem_type(), bool_vec_ty}), Debug("match_merge"));
                for (size_t i = 3; i < split_old->num_args() + 1; i++) {
                    Continuation *next_case_old = nullptr;
                    Continuation *next_case = nullptr;
                    Continuation *next_case_back = nullptr;

                    if (i < split_old->num_args()) {
                        next_case_old = const_cast<Continuation*>(split_old->arg(i)->as<Tuple>()->op(1)->as<Continuation>());
                        next_case = const_cast<Continuation*>(def2def_[next_case_old]->as<Continuation>());
                        next_case_back = world_.continuation(world_.fn_type({world_.mem_type(), bool_vec_ty}), Debug("case_back"));
                    } else {
                        next_case = pre_join;
                        next_case_back = pre_join;
                    }

                    assert(next_case);
                    assert(next_case_back);

                    bool case_back_has_jump = false;

                    for (auto pred_old : join_old->preds()) {
                        if (i != 3 || pred_old != split_old) {
                            if (!div_analysis_->dominatedBy[pred_old].contains(const_cast<Continuation*>(case_old)) && pred_old != case_old) {
                                continue;
                            }
                        }
                        
                        auto pred = const_cast<Continuation*>(def2def_[pred_old]->as<Continuation>());
                        assert(pred);

                        assert(pred->arg(0)->type()->isa<MemType>());
                        const Def * mem = pred->arg(0);
                        assert(mem);
                        assert(is_mem(mem));

                        if (pred_old != split_old) {
                            for (size_t j = 1; j < pred->num_args(); j++) {
                                assert(join_cache[j - 1]);

                                bool predicated = div_analysis_->isPredicated[pred_old];
                                if (predicated) {
                                    const Def* pred_param = pred->param(1);
                                    mem = world_.maskedstore(mem, join_cache[j - 1], pred->arg(j), pred_param);
                                } else {
                                    mem = world_.store(mem, join_cache[j - 1], pred->arg(j));
                                }
                            }

                            pred->jump(case_back, { mem, pred->param(1) });
                        } else {
                            pred->jump(case_back, { mem, pred->arg(1) }); //was a predicated call originally.
                        }

                        auto new_jump_case = world_.predicated(vec_mask);
                        const Def* predicate;
                        if (i < split_old->num_args())
                            predicate = split_predicates[i - 2];
                        else {
                            auto true_elem = world_.literal_bool(true, {});
                            Array<const Def *> elements(vector_width);
                            for (size_t i = 0; i < vector_width; i++) {
                                elements[i] = true_elem;
                            }
                            predicate = world_.vector(elements);
                        }

                        if (next_case == next_case_back) {
                            auto true_elem = world_.literal_bool(true, {});
                            Array<const Def *> elements(vector_width);
                            for (size_t i = 0; i < vector_width; i++) {
                                elements[i] = true_elem;
                            }
                            auto one_predicate = world_.vector(elements);
                            case_back->jump(new_jump_case, { case_back->mem_param(), one_predicate, next_case, next_case_back }, split->debug());
                            case_back_has_jump = true;
                        } else {
                            bool all_one = predicate->isa<Vector>();
                            for (auto op : predicate->ops())
                                if (op != world_.literal_bool(true, {}))
                                    all_one = false;
                            assert(!all_one);
                            case_back->jump(new_jump_case, { case_back->mem_param(), predicate, next_case, next_case_back }, split->debug());
                            case_back_has_jump = true;
                        }
                    }

                    assert(case_back_has_jump);

                    current_case = next_case;
                    case_old = next_case_old;
                    case_back = next_case_back;

                }

                { //mem scope
                    const Def* mem = pre_join->mem_param();

                    Array<const Def*> join_params(join->num_params());
                    for (size_t i = 1; i < join->num_params(); i++) {
                        auto load = world_.load(mem, join_cache[join_is_first_mem ? i - 1 : i]);
                        auto value = world_.extract(load, (int) 1);
                        mem = world_.extract(load, (int) 0);
                        join_params[i] = value;
                    }
                    join_params[0] = mem;
                    pre_join->jump(join, join_params, split->debug());
                }
            }
        }
        //branch:
        else if (cont && cont->is_intrinsic() && cont->intrinsic() == Intrinsic::Branch && latch->arg(1)->type()->isa<VectorType>() && latch->arg(1)->type()->as<VectorType>()->is_vector()) {
            //TODO: Enclosed if-blocks require special care to use the correct predicates throughout execution.
            const Def* current_frame = nullptr;

            if (joins_old.size() > 1) {
                //This should only occur when the condition contains an "and" or "or" instruction.
                //TODO: try to assert this somehow?
                if (joins_old.size() > 2) {
                    world_.dump();
                    for (auto join : joins_old) {
                        join->dump();
                    }
                    std::cerr << "\n";
                    latch_old->dump();
                    std::cerr << "\n";
                }

                /*std::cerr << "cont suc mapping\n";
                for (auto cont : world_.continuations()) {
                    for (auto suc : cont->succs()) {
                        std::cerr << cont->unique_name() << " - " << suc->unique_name() << "\n";
                    }
                }*/
                assert(joins_old.size() == 2 && "Only this case is supported for now.");
            }

#ifdef DUMP_VECTORIZER_LINEARIZER
            std::cerr << "Pre transformation\n";
            DUMP_BLOCK(latch_old);
            DUMP_BLOCK(latch);
#endif

#ifdef DUMP_VECTORIZER_LINEARIZER
            std::cerr << "PreAll\n";
            DUMP_BLOCK(latch);
#endif

            //Check that all join nodes already take a mem parameter.
            for (auto join_old : joins_old) {
                auto join_new = def2def_[join_old]->as<Continuation>();
                assert(is_mem(join_new->param(0)));
            }

            GIDMap<const Continuation*, Array<const Def *>> join_caches(joins_old.size());
            size_t num_enclosed_splits = encloses_splits[latch_old].size();
            assert(num_enclosed_splits <= 1 && "There should only  be one enclosed split right now: then_new");
            GIDMap<const Continuation*, const Def *> predicate_caches(num_enclosed_splits ? num_enclosed_splits : 1); //Size must not be 0.

            { //mem scope
                assert(latch->arg(0)->type()->isa<MemType>());
                const Def * mem = latch->arg(0);
                assert(mem);
                assert(is_mem(mem));

                for (auto join_old : joins_old) {
                    auto join = def2def_[join_old]->as<Continuation>();
                    size_t cache_size = join->num_params() - 1;
                    join_caches[join] = Array<const Def *>(cache_size);

                    if (cache_size && !current_frame) {
                        auto mem = vectorized->mem_param();
                        if (!enter)
                            enter = world_.enter(mem)->as<Enter>();
                        current_frame = enter->out_frame();
                        auto newmem = enter->out_mem();
                        for (auto use : mem->copy_uses()) {
                            auto use_inst = use.def();
                            if (use_inst == enter)
                                continue;
                            assert(use_inst);
                            int index = use.index();
                            Def* olduse = const_cast<Def*>(use_inst);
                            olduse->unset_op(index);
                            olduse->set_op(index, newmem);
                        }
                    }

                    for (size_t i = 0; i < cache_size; i++) {
                        assert(current_frame);
                        auto t = world_.slot(join->param(i + 1)->type(), current_frame, Debug("join_cache_branch"));
                        join_caches[join][i] = t;
                    }
                }

                if (num_enclosed_splits) {
                    if (!current_frame) {
                        auto mem = vectorized->mem_param();
                        if (!enter)
                            enter = world_.enter(mem)->as<Enter>();
                        current_frame = enter->out_frame();
                        auto newmem = enter->out_mem();
                        for (auto use : mem->copy_uses()) {
                            auto use_inst = use.def();
                            if (use_inst == enter)
                                continue;
                            assert(use_inst);
                            int index = use.index();
                            Def* olduse = const_cast<Def*>(use_inst);
                            olduse->unset_op(index);
                            olduse->set_op(index, newmem);
                        }
                    }

                    auto false_elem = world_.literal_bool(false, {});
                    Array<const Def *> elements(vector_width);
                    for (size_t i = 0; i < vector_width; i++)
                        elements[i] = false_elem;
                    auto zero_predicate = world_.vector(elements);
                    for (auto split_old : encloses_splits[latch_old]) {
                        const Continuation* split_new = def2def_[split_old]->as<Continuation>();
                        assert(split_new);
                        assert(current_frame);
                        auto pred_cache = world_.slot(zero_predicate->type(), current_frame, Debug("predicate_cache"));
                        mem = world_.store(mem, pred_cache, zero_predicate);
                        predicate_caches[split_new] = pred_cache;
                    }
                }

                //TODO: This new mem should replace all old mem instances!
            }

            GIDMap<const Continuation*, const Continuation *> pre_joins(joins_old.size());
            for (auto join_old : joins_old) {
                //join == some_latch can occur at times!

                //TODO: Does join take a predicate parameter? It probably should.
                auto join = def2def_[join_old]->as<Continuation>();
                auto bool_vec_ty = widen(world_.type_bool());
                Continuation * pre_join = world_.continuation(world_.fn_type({world_.mem_type(), bool_vec_ty}), Debug("branch_merge"));

                const Def* mem = pre_join->param(0);
                Array<const Def*> join_params(join->num_params());
                auto &join_cache = join_caches[join];

                for (size_t i = 1; i < join->num_params(); i++) {
                    auto load = world_.load(mem, join_cache[i - 1]);
                    auto value = world_.extract(load, (int) 1);
                    mem = world_.extract(load, (int) 0);
                    join_params[i] = value;
                }
                join_params[0] = mem;
                pre_join->jump(join, join_params, latch->debug());

                pre_joins[join] = pre_join;
            }

            //TODO: This code still feels incomplete and broken. Devise a better way to handle this rewiring of the mem-monad.
            //Maybe I should extend all cases surrounding a latch with a memory operand with code similar to schedule verification.
            //I depend on the schedule from time to time. This is not good.
            Array<const Continuation*> splits(encloses_splits[latch_old].size()+1);
            splits[0] = latch_old;
            size_t i = 1;
            for (auto split_old : encloses_splits[latch_old])
                splits[i++] = split_old;
            std::sort (splits.begin(), splits.end(), [&](const Continuation *ac, const Continuation *bc) {
                    Continuation *a = const_cast<Continuation*>(ac);
                    Continuation *b = const_cast<Continuation*>(bc);
                    return div_analysis_->dominatedBy[a].contains(b);
                });

            auto vec_mask = widen(world_.type_bool());

            GIDMap<const Continuation*, const Continuation*> rewired_predecessors;

#ifdef DUMP_VECTORIZER_LINEARIZER
            std::cerr << "Before rewiring\n";
            DUMP_BLOCK(vectorized);

            for (size_t current_split_index = 0; current_split_index < splits.size(); current_split_index++) {
                auto split_old = splits[current_split_index];
                auto split = const_cast<Continuation*>(def2def_[split_old]->as<Continuation>());
                assert(split);
            }
#endif

            auto bool_vec_ty = widen(world_.type_bool());

            for (size_t current_split_index = 0; current_split_index < splits.size(); current_split_index++) {
                auto split_old = splits[current_split_index];
                auto split = const_cast<Continuation*>(def2def_[split_old]->as<Continuation>());
                assert(split);

                const Continuation * then_old = split_old->arg(2)->as<Continuation>();
                Continuation * then_new = const_cast<Continuation*>(def2def_[then_old]->as<Continuation>());
                assert(then_new);

                if (then_new->num_params() == 1) {
                    Continuation * predicated_buffer = world_.continuation(world_.fn_type({world_.mem_type(), bool_vec_ty}));
                    predicated_buffer->jump(then_new, {predicated_buffer->param(0)});
                    then_new = predicated_buffer;
                }

                const Continuation * else_old = split_old->arg(3)->as<Continuation>();
                Continuation * else_new = const_cast<Continuation*>(def2def_[else_old]->as<Continuation>());
                assert(else_new);

                if (else_new->num_params() == 1) {
                    Continuation * predicated_buffer = world_.continuation(world_.fn_type({world_.mem_type(), bool_vec_ty}));
                    predicated_buffer->jump(else_new, {predicated_buffer->param(0)});
                    else_new = predicated_buffer;
                }

                //TODO: These predicates need to be extended to include the cached predicates as well.
                //TODO: We need to store predicates as well!
                const Def* predicate_true = split->arg(1);

                assert(predicate_true);
                if (!predicate_true->type()->isa<VectorType>() || !predicate_true->type()->as<VectorType>()->is_vector()) {
                    Array<const Def *> elements(vector_width);
                    for (size_t i = 0; i < vector_width; i++) {
                        elements[i] = predicate_true;
                    }
                    predicate_true = world_.vector(elements);
                }
                assert(predicate_true->type()->isa<VectorType>() && predicate_true->type()->as<VectorType>()->is_vector());

                bool all_one = predicate_true->isa<Vector>();
                if (all_one)
                    for (auto op : predicate_true->ops())
                        if (op != world_.literal_bool(true, {}))
                            all_one = false;
                if (all_one) {
                    predicate_true->dump();
                    predicate_true->type()->dump();
                    split->dump();
                    std::cerr << "\n";
                }
                assert(!all_one);

                const Def* predicate_false = world_.arithop_not(predicate_true);
                assert(predicate_false);

                all_one = predicate_false->isa<Vector>();
                if (all_one)
                    for (auto op : predicate_false->ops())
                        if (op != world_.literal_bool(true, {}))
                            all_one = false;
                assert(!all_one);

                Continuation * then_new_back = world_.continuation(world_.fn_type({world_.mem_type(), bool_vec_ty}), Debug("branch_true_back"));
                Continuation * else_new_back = world_.continuation(world_.fn_type({world_.mem_type(), bool_vec_ty}), Debug("branch_false_back"));

                auto new_jump_latch = world_.predicated(vec_mask);

                { //mem scope
                    assert(split->arg(0)->type()->isa<MemType>());
                    const Def* mem = split->arg(0);
                    assert(mem);
                    assert(is_mem(mem));

                    assert(then_new != then_new_back);
                    split->jump(new_jump_latch, { mem, predicate_true, then_new, then_new_back }, latch->debug());
                    rewired_predecessors.emplace(split, nullptr);
                }

                //Connect then-nodes to then-back
                //If there are loops present on the then-side, we will not find an appropriate join node for that and the then_back node will be executed once all vector elements are done executing the loop.
                //TODO: There might be an issue: There might be loops with the exit being the "then" case, then and else should be switched then!
                for (auto join_old : joins_old) {
                    auto join = const_cast<Continuation*>(def2def_[join_old]->as<Continuation>());
                    assert(join);
                    auto &join_cache = join_caches[join];

                    for (auto pred_old : join_old->preds()) {
                        if (!div_analysis_->dominatedBy[pred_old].contains(const_cast<Continuation*>(then_old)) && pred_old != then_old)
                            continue;
                        auto pred = const_cast<Continuation*>(def2def_[pred_old]->as<Continuation>());
                        if (rewired_predecessors.contains(pred))
                            if (rewired_predecessors[pred] == nullptr)
                                continue;

                        //get old mem parameter, if possible.
                        assert(pred->arg(0)->type()->isa<MemType>());
                        const Def* mem = pred->arg(0);
                        assert(mem);
                        assert(is_mem(mem));

                        if (join_cache.size() < pred->num_args() - 1) {
                            std::cerr << "Error\n";
                            DUMP_BLOCK(split);
                            std::cerr << "pred\n";
                            DUMP_BLOCK(pred);
                        }

                        for (size_t j = 1; j < pred->num_args(); j++) {
                            assert(join_cache[j - 1]);

                            bool predicated = div_analysis_->isPredicated[pred_old];
                            if (predicated) {
                                const Def* pred_param = pred->param(1);
                                mem = world_.maskedstore(mem, join_cache[j - 1], pred->arg(j), pred_param);
                            } else {
                                mem = world_.store(mem, join_cache[j - 1], pred->arg(j));
                            }
                        }

                        pred->jump(then_new_back, { mem, pred->param(1) });
                        rewired_predecessors.emplace(pred, nullptr);
                    }
                }


                //connect then-back to else-branch
                auto new_jump_then = world_.predicated(vec_mask);
                assert(else_new != else_new_back);
                then_new_back->jump(new_jump_then, { then_new_back->mem_param(), predicate_false, else_new, else_new_back }, latch->debug());

                Continuation *else_join_cache = nullptr;

                //Connect else-nodes to else-back
                for (auto join_old : joins_old) {
                    auto join = const_cast<Continuation*>(def2def_[join_old]->as<Continuation>());
                    assert(join);
                    auto &join_cache = join_caches[join];

                    for (auto pred_old : join_old->preds()) {
                        if (!div_analysis_->dominatedBy[pred_old].contains(const_cast<Continuation*>(else_old)) && pred_old != else_old)
                            continue;
                        auto pred = const_cast<Continuation*>(def2def_[pred_old]->as<Continuation>());
                        if (rewired_predecessors.contains(pred))
                            if (rewired_predecessors[pred] == nullptr)
                                continue;

                        assert(!else_join_cache);
                        else_join_cache = join;

                        //get old mem parameter, if possible.
                        assert(pred->arg(0)->type()->isa<MemType>());
                        const Def* mem = pred->arg(0);
                        assert(mem);
                        assert(is_mem(mem));

                        for (size_t j = 1; j < pred->num_args(); j++) {
                            assert(join_cache[j - 1]);
                            bool predicated = div_analysis_->isPredicated[pred_old];
                            if (predicated) {
                                const Def* pred_param = pred->param(1);
                                mem = world_.maskedstore(mem, join_cache[j - 1], pred->arg(j), pred_param);
                            } else {
                                mem = world_.store(mem, join_cache[j - 1], pred->arg(j));
                            }
                        }

                        auto pre_join = pre_joins[join];
                        assert(pre_join);

                        pred->jump(else_new_back, { mem, pred->param(1) });
                        rewired_predecessors.emplace(pred, pre_join);
                    }
                }

                auto true_elem = world_.literal_bool(true, {});
                Array<const Def *> elements(vector_width);
                for (size_t i = 0; i < vector_width; i++) {
                    elements[i] = true_elem;
                }
                auto one_predicate = world_.vector(elements);

                auto new_jump_else = world_.predicated(vec_mask);

                if (!else_join_cache) {
                    //There is no join after else. This should only occur with loops, and else_back should not be reachable in this case.
                    //TODO: This should also imply that the current mask is no fully populated, as the else-block is only reachable once all loop instances are done executing.
                    
                    //else_new_back->jump(else_new->op(0), { else_new_back->mem_param() });
                    const Continuation * return_cont = else_new->op(0)->isa_continuation();
                    if (return_cont->num_params() == 1) {
                        Continuation * predicated_buffer = world_.continuation(world_.fn_type({world_.mem_type(), bool_vec_ty}));
                        predicated_buffer->jump(return_cont, {predicated_buffer->param(0)});
                        return_cont = predicated_buffer;
                    }

                    else_new_back->jump(new_jump_else, { else_new_back->mem_param(), one_predicate, return_cont, return_cont}, latch->debug()); //TODO: This is only correct if the target is a return.
                } else {
                    //connect else-back to the cached join.
                    
                    auto pre_join = pre_joins[else_join_cache];
                    assert(pre_join);
                    else_new_back->jump(new_jump_else, { else_new_back->mem_param(), one_predicate, pre_join, pre_join }, latch->debug());
                }

                //std::cerr << "After rewiring " << current_split_index + 1 << "\n";
                //rewired_predecessors.dump();
                //Scope(vectorized).dump();
            }
        } else {
            THORIN_UNREACHABLE;
        }

#ifdef DUMP_VECTORIZER_LINEARIZER
        DUMP_BLOCK(vectorized);
#endif
    }

#ifdef DUMP_VECTORIZER_LINEARIZER
    std::cerr << "Post lin\n";
    DUMP_BLOCK(vectorized);
#endif

}


bool Vectorizer::run() {
#ifdef DUMP_VECTORIZER
    world_.dump();
#endif

    DBG_TIME(vregion, time);

    for (auto continuation : world_.exported_continuations()) {
        enqueue(continuation);
    }

    //Task 1: Divergence Analysis
    //Task 1.1: Find all vectorization continuations
    while (!queue_.empty()) {
        Continuation *cont = pop(queue_);

        if (cont->intrinsic() == Intrinsic::Vectorize) {
#ifdef DUMP_VECTORIZER
            std::cerr << "Continuation\n";
            cont->dump();
#endif

            for (auto pred : cont->preds()) {
                auto *kernarg = dynamic_cast<const Global *>(pred->arg(2));
                assert(kernarg && "Not sure if all kernels are always declared globally");
                assert(!kernarg->is_mutable() && "Self transforming code is not supported here!");
                auto *kerndef = kernarg->init()->isa_continuation();
                assert(kerndef && "We need a continuation for vectorization");

    //Task 1.2: Divergence Analysis for each vectorize block
    //Task 1.2.1: Ensure the return intrinsic is only called once, to make the job of the divergence-analysis easier.
                auto ret_param = kerndef->ret_param();
#ifdef DUMP_VECTORIZER
                ret_param->dump();
#endif
                //if (ret_param->uses().size() > 1) {
                {
                    auto ret_type = const_cast<FnType*>(ret_param->type()->as<FnType>());
                    Continuation * ret_join = world_.continuation(ret_type, Debug("shim"));
                    ret_param->replace(ret_join);

                    Array<const Def*> args(ret_join->num_params());
                    for (size_t i = 0, e = ret_join->num_params(); i < e; i++) {
                        args[i] = ret_join->param(i);
                    }

                    ret_join->jump(ret_param, args);
#ifdef DUMP_VECTORIZER
                    DUMP_BLOCK(kerndef);
#endif
                }

                {
                    auto mem_cleanup = new MemCleaner(kerndef);
                    mem_cleanup->run();
                }

    //Warning: Will fail to produce meaningful results or rightout break the program if kerndef does not dominate its subprogram
                {
                    DBG_TIME(div_time, time_div);
                    div_analysis_ = new DivergenceAnalysis(kerndef);
                    div_analysis_->run();
                }

#ifdef DUMP_VECTORIZER
                div_analysis_->dump();
#endif

    //Task 2: Widening
                Continuation* vectorized;
                {
                    DBG_TIME(widen_time, time_widen);
                    widen_setup(kerndef);
                    vectorized = widen();
                }
                //auto *vectorized = clone(Scope(kerndef));
                assert(vectorized);
                def2def_[kerndef] = vectorized;

                //DUMP_BLOCK(vectorized);

    //Task 3: Linearize divergent controll flow
                {
                    DBG_TIME(lin_time, time_lin);

                    linearize(vectorized);
                }

                //DUMP_BLOCK(vectorized);

                delete div_analysis_;

#ifdef DUMP_VECTORIZER
                world_.dump();
                std::cerr << "cont suc mapping 2\n";
                for (auto cont : world_.continuations()) {
                    for (auto suc : cont->succs()) {
                        std::cerr << cont->unique_name() << " - " << suc->unique_name() << "\n";
                    }
                }
#endif

    //Task 4: Rewrite vectorize call
                if (vectorized) {
                    for (auto caller : cont->preds()) {
                        Array<const Def*> args(vectorized->num_params());

                        args[0] = caller->arg(0); //mem
                        //args[1] = caller->arg(1); //width
                        Array<const Def*> defs(vector_width);
                        for (size_t i = 0; i < vector_width; i++) {
                            defs[i] = world_.literal_qs32(i, caller->arg(1)->debug_history());
                        }
                        args[1] = world_.vector(defs, caller->arg(1)->debug_history());

                        for (size_t p = 2; p < vectorized->num_params(); p++) {
                            args[p] = caller->arg(p + 1);
                        }

                        caller->jump(vectorized, args, caller->debug());
                    }
                }
            }
#ifdef DUMP_VECTORIZER
            std::cerr << "Continuation end\n\n";
#endif
        }

        for (auto succ : cont->succs())
            enqueue(succ);
    }

#ifdef DUMP_VECTORIZER
    std::cout << "End vectorizer\n";
    world_.dump();
#endif

    return false;
}

bool vectorize(World& world) {
    world.VLOG("start vectorizer");
    bool res = Vectorizer(world).run();

    if (!res)
        flatten_vectors(world);
    world.cleanup();

    debug_verify(world);

    world.VLOG("end vectorizer");
    return res;
}

}
