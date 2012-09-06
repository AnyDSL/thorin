#include "anydsl/be/llvm.h"

#include <boost/unordered_map.hpp>

#include <llvm/Constant.h>
#include <llvm/Constants.h>
#include <llvm/Function.h>
#include <llvm/LLVMContext.h>
#include <llvm/Module.h>
#include <llvm/Type.h>
#include <llvm/Analysis/Verifier.h>
#include <llvm/Support/IRBuilder.h>

#include "anydsl/def.h"
#include "anydsl/lambda.h"
#include "anydsl/literal.h"
#include "anydsl/primop.h"
#include "anydsl/type.h"
#include "anydsl/world.h"
#include "anydsl/analyses/rootlambdas.h"
#include "anydsl/analyses/domtree.h"
#include "anydsl/analyses/placement.h"
#include "anydsl/util/array.h"

namespace anydsl {
namespace be_llvm {

//------------------------------------------------------------------------------

typedef boost::unordered_map<const Lambda*, llvm::Function*> FctMap;
typedef boost::unordered_map<const Lambda*, llvm::BasicBlock*> BBMap;
typedef boost::unordered_map<const Param*, llvm::Value*> ParamMap;
typedef boost::unordered_map<const Param*, llvm::PHINode*> PhiMap;
typedef boost::unordered_map<const PrimOp*, llvm::Value*> PrimOpMap;

//------------------------------------------------------------------------------

class CodeGen {
public:

    CodeGen(const World& world);

    void emit();

    llvm::Type* convert(const Type* type);
    llvm::Value* emit(const Def* def);
    llvm::Value* literal(const PrimLit* def);
    llvm::Value* lookup(const Def* def);

private:

    const World& world_;
    llvm::LLVMContext context_;
    llvm::IRBuilder<> builder_;
    llvm::Module* module_;
    ParamMap params_;
    PhiMap phis_;
    PrimOpMap primops_;
};

CodeGen::CodeGen(const World& world)
    : world_(world)
    , context_()
    , builder_(context_)
    , module_(new llvm::Module("anydsl", context_))
{}

//------------------------------------------------------------------------------

void CodeGen::emit() {
    LambdaSet roots = find_root_lambdas(world_.lambdas());

    FctMap fcts;

    // map all root-level lambdas to llvm function stubs
    for_all (lambda, roots) {
        llvm::FunctionType* ft = llvm::cast<llvm::FunctionType>(convert(lambda->type()));
        llvm::Function* f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, lambda->debug, module_);
        fcts.insert(std::make_pair(lambda, f));
    }

    // for all top-level functions
    for_all (lf, fcts) {
        const Lambda* lambda = lf.first;
        llvm::Function* fct = lf.second;
        const Param* ret_param;

        llvm::Function::arg_iterator arg = fct->arg_begin();
        size_t i = 0;
        
        // map params
        for_all (param, lambda->params()) {
            while (i < param->index()) {
                ++i;
                ++arg;
            }

            if (param->type()->isa<Pi>())
                ret_param = param;
            else {
                params_[param] = arg;
                arg->setName(param->debug);
                ++arg;
            }

            ++i;
        }

        DomTree domtree = calc_domtree(lambda);
        BBMap bbs;

        // map all bb-like lambdas to llvm bb stubs
        for_all (node, domtree.bfs())
            bbs[node->lambda()] = llvm::BasicBlock::Create(context_, node->lambda()->debug, fct);

        // create phi node stubs (for all nodes except entry)
        for_all (node, domtree.bfs().slice_back(1)) {
            const Lambda* lambda = node->lambda();

            // deal with special call-lambda-cascade:
            // lambda(...) jump (foo, [..., lambda(...) ..., ...]
            if (lambda->uses().size() == 1) {
                Use use = *lambda->uses().begin();
                if (use.def()->isa<Lambda>() && use.index() > 0)
                    continue;// do not register a phi, see also case 1c) in 'terminate bb' below
            }

            builder_.SetInsertPoint(bbs[lambda]);

            for_all (param, node->lambda()->params())
                phis_[param] = builder_.CreatePHI(convert(param->type()), param->phi().size(), param->debug);
        }

        Array< std::vector<const PrimOp*> > places = place(domtree);

        // emit body for each bb
        for_all (node, domtree.bfs()) {
            const Lambda* lambda = node->lambda();
            builder_.SetInsertPoint(bbs[node->lambda()]);
            std::vector<const PrimOp*> primops = places[node->index()];

            for_all (primop, primops) {
                if (!primop->type()->isa<Pi>()) // don't touch higher-order primops
                    primops_[primop] = emit(primop);
            }

            // terminate bb
            switch (lambda->targets().size()) {
                case 0: { 
                    // this is a return
                    const Param* param = lambda->to()->as<Param>();
                    assert(param == ret_param);
                    assert(lambda->args().size() == 1);
                    builder_.CreateRet(lookup(lambda->arg(0)));
                    break;
                } 
                case 1: { 
                    // three subcases
                    const Lambda* tolambda = lambda->to()->as<Lambda>();
                    const Pi* topi = tolambda->pi();
                    size_t ho = tolambda->pi()->ho_begin();

                    if (ho == tolambda->pi()->ho_end()) // case a) ordinary jump
                        builder_.CreateBr(bbs[tolambda]);
                    else {
                        // put all first-order args into array
                        Array<llvm::Value*> args(lambda->args().size() - 1);

                        for (size_t i = topi->fo_begin(), e = topi->fo_end(), j = 0; i != e; topi->fo_next(i))
                            args[j++] = lookup(lambda->arg(i));

                        llvm::ArrayRef<llvm::Value*> ref(args.begin(), args.end());
                        llvm::CallInst* call = builder_.CreateCall(fcts[tolambda], ref);
                        
                        const Def* ho_arg = lambda->arg(ho);
                        if (ho_arg == ret_param)        // case b) call + return
                            builder_.CreateRet(call); 
                        else {                          // case c) call + continuation
                            const Lambda* succ = lambda->arg(ho)->as<Lambda>();
                            const Param* ho_param = succ->param(0);
                            if (ho_param)
                                params_[ho_param] = call;

                            builder_.CreateBr(bbs[succ]);
                        }
                    }
                    break;
                } 
                case 2: { 
                    // this is a branch
                    const Select* select = lambda->to()->as<Select>();
                    const Lambda* tlambda = select->tval()->as<Lambda>();
                    const Lambda* flambda = select->fval()->as<Lambda>();

                    llvm::Value* cond = lookup(select->cond());
                    llvm::BasicBlock* tbb = bbs[tlambda];
                    llvm::BasicBlock* fbb = bbs[flambda];
                    builder_.CreateCondBr(cond, tbb, fbb);
                    break;
                } 
                default:
                    ANYDSL_UNREACHABLE;
            }
        }

        // add missing arguments to phis
        for_all (p, phis_) {
            const Param* param = p.first;
            llvm::PHINode* phi = p.second;

            for_all (op, param->phi())
                phi->addIncoming(lookup(op.def()), bbs[op.from()]);
        }

        params_.clear();
        phis_.clear();
        primops_.clear();
    }

    module_->dump();
    llvm::verifyModule(*this->module_);
    delete module_;
}

llvm::Value* CodeGen::lookup(const Def* def) {
    if (const PrimLit* lit = def->isa<PrimLit>())
        return literal(lit);

    if (const PrimOp* primop = def->isa<PrimOp>())
        return primops_[primop];

    const Param* param = def->as<Param>();
    ParamMap::iterator i = params_.find(param);
    if (i != params_.end())
        return i->second;

    assert(phis_.find(param) != phis_.end());

    return phis_[param];
}

llvm::Value* CodeGen::literal(const PrimLit* lit) {
    llvm::Type* type = convert(lit->type());
    Box box = lit->box();

    switch (lit->primtype_kind()) {
        case PrimType_u1:  return builder_.getInt1(box.get_u1().get());
        case PrimType_u8:  return builder_.getInt8(box.get_u8());
        case PrimType_u16: return builder_.getInt16(box.get_u16());
        case PrimType_u32: return builder_.getInt32(box.get_u32());
        case PrimType_u64: return builder_.getInt64(box.get_u64());
        case PrimType_f32: return llvm::ConstantFP::get(type, box.get_f32());
        case PrimType_f64: return llvm::ConstantFP::get(type, box.get_f64());
        default: ANYDSL_UNREACHABLE;
    }
}

llvm::Value* CodeGen::emit(const Def* def) {
    if (const BinOp* bin = def->isa<BinOp>()) {
        llvm::Value* lhs = lookup(bin->lhs());
        llvm::Value* rhs = lookup(bin->rhs());

        if (const RelOp* rel = def->isa<RelOp>()) {
            switch (rel->relop_kind()) {
                case RelOp_cmp_eq:   return builder_.CreateICmpEQ (lhs, rhs);
                case RelOp_cmp_ne:   return builder_.CreateICmpNE (lhs, rhs);

                case RelOp_cmp_ugt:  return builder_.CreateICmpUGT(lhs, rhs);
                case RelOp_cmp_uge:  return builder_.CreateICmpUGE(lhs, rhs);
                case RelOp_cmp_ult:  return builder_.CreateICmpULT(lhs, rhs);
                case RelOp_cmp_ule:  return builder_.CreateICmpULE(lhs, rhs);

                case RelOp_cmp_sgt:  return builder_.CreateICmpSGT(lhs, rhs);
                case RelOp_cmp_sge:  return builder_.CreateICmpSGE(lhs, rhs);
                case RelOp_cmp_slt:  return builder_.CreateICmpSLT(lhs, rhs);
                case RelOp_cmp_sle:  return builder_.CreateICmpSLE(lhs, rhs);

                case RelOp_fcmp_oeq: return builder_.CreateFCmpOEQ(lhs, rhs);
                case RelOp_fcmp_one: return builder_.CreateFCmpONE(lhs, rhs);

                case RelOp_fcmp_ogt: return builder_.CreateFCmpOGT(lhs, rhs);
                case RelOp_fcmp_oge: return builder_.CreateFCmpOGE(lhs, rhs);
                case RelOp_fcmp_olt: return builder_.CreateFCmpOLT(lhs, rhs);
                case RelOp_fcmp_ole: return builder_.CreateFCmpOLE(lhs, rhs);

                case RelOp_fcmp_ueq: return builder_.CreateFCmpUEQ(lhs, rhs);
                case RelOp_fcmp_une: return builder_.CreateFCmpUNE(lhs, rhs);

                case RelOp_fcmp_ugt: return builder_.CreateFCmpUGT(lhs, rhs);
                case RelOp_fcmp_uge: return builder_.CreateFCmpUGE(lhs, rhs);
                case RelOp_fcmp_ult: return builder_.CreateFCmpULT(lhs, rhs);
                case RelOp_fcmp_ule: return builder_.CreateFCmpULE(lhs, rhs);

                case RelOp_fcmp_uno: return builder_.CreateFCmpUNO(lhs, rhs);
                case RelOp_fcmp_ord: return builder_.CreateFCmpORD(lhs, rhs);
            }
        }

        const ArithOp* arith = def->as<ArithOp>();

        switch (arith->arithop_kind()) {
            case ArithOp_add:  return builder_.CreateAdd (lhs, rhs);
            case ArithOp_sub:  return builder_.CreateSub (lhs, rhs);
            case ArithOp_mul:  return builder_.CreateMul (lhs, rhs);
            case ArithOp_udiv: return builder_.CreateUDiv(lhs, rhs);
            case ArithOp_sdiv: return builder_.CreateSDiv(lhs, rhs);
            case ArithOp_urem: return builder_.CreateURem(lhs, rhs);
            case ArithOp_srem: return builder_.CreateSRem(lhs, rhs);

            case ArithOp_fadd: return builder_.CreateFAdd(lhs, rhs);
            case ArithOp_fsub: return builder_.CreateFSub(lhs, rhs);
            case ArithOp_fmul: return builder_.CreateFMul(lhs, rhs);
            case ArithOp_fdiv: return builder_.CreateFDiv(lhs, rhs);
            case ArithOp_frem: return builder_.CreateFRem(lhs, rhs);

            case ArithOp_and:  return builder_.CreateAnd (lhs, rhs);
            case ArithOp_or:   return builder_.CreateOr  (lhs, rhs);
            case ArithOp_xor:  return builder_.CreateXor (lhs, rhs);

            case ArithOp_shl:  return builder_.CreateShl (lhs, rhs);
            case ArithOp_lshr: return builder_.CreateLShr(lhs, rhs);
            case ArithOp_ashr: return builder_.CreateAShr(lhs, rhs);
        }
    }

    if (const ConvOp* conv = def->isa<ConvOp>()) {
        llvm::Value* from = lookup(conv->from());
        llvm::Type* to = convert(conv->type());

        switch (conv->convop_kind()) {
            case ConvOp_trunc:  return builder_.CreateTrunc  (from, to);
            case ConvOp_zext:   return builder_.CreateZExt   (from, to);
            case ConvOp_sext:   return builder_.CreateSExt   (from, to);
            case ConvOp_stof:   return builder_.CreateSIToFP (from, to);
            case ConvOp_utof:   return builder_.CreateSIToFP (from, to);
            case ConvOp_ftrunc: return builder_.CreateFPTrunc(from, to);
            case ConvOp_ftos:   return builder_.CreateFPToSI (from, to);
            case ConvOp_ftou:   return builder_.CreateFPToUI (from, to);
            case ConvOp_fext:   return builder_.CreateFPExt  (from, to);
            case ConvOp_bitcast:return builder_.CreateBitCast(from, to);
        }
    }

    if (const Select* select = def->isa<Select>()) {
        llvm::Value* cond = lookup(select->cond());
        llvm::Value* tval = lookup(select->tval());
        llvm::Value* fval = lookup(select->fval());
        return builder_.CreateSelect(cond, tval, fval);
    }

    if (const TupleOp* tupleop = def->isa<TupleOp>()) {
        llvm::Value* tuple = lookup(tupleop->tuple());
        unsigned idxs[1] = { unsigned(tupleop->index()) };

        if (tupleop->node_kind() == Node_Extract)
            return builder_.CreateExtractValue(tuple, idxs);

        const Insert* insert = def->as<Insert>();
        llvm::Value* value = lookup(insert->value());

        return builder_.CreateInsertValue(tuple, value, idxs);
    }

    if (const Tuple* tuple = def->isa<Tuple>()) {
        llvm::Value* agg = llvm::UndefValue::get(convert(tuple->type()));

        for (unsigned i = 0, e = tuple->ops().size(); i != e; ++i) {
            unsigned idxs[1] = { unsigned(i) };
            agg = builder_.CreateInsertValue(agg, lookup(tuple->op(i)), idxs);
        }

        return agg;
    }

    // bottom and any
    if (const Undef* undef = def->isa<Undef>())
        return llvm::UndefValue::get(convert(undef->type()));

    def->dump();
    ANYDSL_UNREACHABLE;
}

llvm::Type* CodeGen::convert(const Type* type) {
    switch (type->node_kind()) {
        case Node_PrimType_u1:  return llvm::IntegerType::get(context_, 1);
        case Node_PrimType_u8:  return llvm::IntegerType::get(context_, 8);
        case Node_PrimType_u16: return llvm::IntegerType::get(context_, 16);
        case Node_PrimType_u32: return llvm::IntegerType::get(context_, 32);
        case Node_PrimType_u64: return llvm::IntegerType::get(context_, 64);
        case Node_PrimType_f32: return llvm::Type::getFloatTy(context_);
        case Node_PrimType_f64: return llvm::Type::getDoubleTy(context_);

        case Node_Pi: {
            // extract "return" type, collect all other types
            const Pi* pi = type->as<Pi>();
            llvm::Type* retType = 0;
            size_t i = 0;
            Array<llvm::Type*> elems(pi->size() - 1);

            for_all (elem, pi->elems()) {
                if (const Pi* pi = elem->isa<Pi>()) {
                    switch (pi->size()) {
                        case 0:
                            retType = llvm::Type::getVoidTy(context_);
                            break;
                        case 1:
                            retType = convert(pi->elem(0));
                            break;
                        default: {
                            Array<llvm::Type*> elems(pi->size());
                            size_t i = 0;
                            for_all (elem, pi->elems())
                                elems[i++] = convert(elem);

                            llvm::ArrayRef<llvm::Type*> structTypes(elems.begin(), elems.end());
                            retType = llvm::StructType::get(context_, structTypes);
                            break;
                        }
                    }
                } else
                    elems[i++] = convert(elem);
            }

            assert(retType);
            llvm::ArrayRef<llvm::Type*> paramTypes(elems.begin(), elems.end());
            return llvm::FunctionType::get(retType, paramTypes, false);
        }

        case Node_Sigma: {
            // TODO watch out for cycles!

            const Sigma* sigma = type->as<Sigma>();

            Array<llvm::Type*> elems(sigma->elems().size());
            size_t i = 0;
            for_all (t, sigma->elems())
                elems[i++] = convert(t);

            return llvm::StructType::get(context_, llvm::ArrayRef<llvm::Type*>(elems.begin(), elems.end()));
        }

        default: ANYDSL_UNREACHABLE;
    }
}

//------------------------------------------------------------------------------

void emit(const World& world) {
    CodeGen cg(world);
    cg.emit();
}

//------------------------------------------------------------------------------

} // namespace anydsl
} // namespace be_llvm
