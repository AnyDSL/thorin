#ifdef LLVM_SUPPORT

#include "thorin/be/llvm.h"

#include <algorithm>
#include <iostream>
#include <unordered_map>

#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Analysis/Verifier.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include "thorin/def.h"
#include "thorin/lambda.h"
#include "thorin/literal.h"
#include "thorin/memop.h"
#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/array.h"

#ifdef WFV2_SUPPORT
#include <wfvInterface.h>
#endif

namespace thorin {

template<class T> 
llvm::ArrayRef<T> llvm_ref(const Array<T>& array) { return llvm::ArrayRef<T>(array.begin(), array.end()); }

//------------------------------------------------------------------------------

typedef LambdaMap<llvm::BasicBlock*> BBMap;

class CodeGen {
public:
    CodeGen(World& world, EmitHook& hook);

    void emit_cuda_decls();
    void emit_vector_decls();
    void emit_cuda(Lambda* target, BBMap& bbs);
    void emit_vectors(llvm::Function* current, Lambda* target, BBMap& bbs);
    void emit();
    void postprocess();
    void dump();
    llvm::Type* map(const Type* type);
    llvm::Value* emit(Def def);
    llvm::Value* lookup(Def def);

private:
    World& world;
    EmitHook& hook;
    llvm::LLVMContext context;
    llvm::IRBuilder<> builder;
    AutoPtr<llvm::Module> module;
    AutoPtr<llvm::Module> cuda_module;
    std::unordered_map<const Param*, llvm::Value*> params;
    std::unordered_map<const Param*, llvm::PHINode*> phis;
    std::unordered_map<const PrimOp*, llvm::Value*> primops;
    std::unordered_map<Lambda*, llvm::Function*> fcts;

    // vectors
    llvm::Type* vector_tid_type;
    llvm::Function* vector_tid_getter;
    struct VectorizationEntry {
        llvm::PHINode* loop_counter;
        llvm::CallInst* kernel_call;
        llvm::Function* func;
        llvm::Function* kernel_func;
        llvm::Function* kernel_simd_func;
        u32 vector_length;
    };
    std::vector<VectorizationEntry> v_fcts;

    // cuda
    llvm::Function* cuda_thread_id_getter[3];
    llvm::Function* cuda_block_id_getter[3];
    llvm::Function* cuda_block_dim_getter[3];
    llvm::Function* malloc_memory;
    llvm::Function* write_memory;
    llvm::Function* write_memory_indir;
    llvm::Function* load_kernel;
    llvm::Function* set_kernel_arg;
    llvm::Function* set_problem_size;
    llvm::Function* launch_kernel;
    llvm::Function* synchronize;
    llvm::Function* read_memory;
    llvm::Function* read_memory_indir;
    llvm::Function* free_memory;
    llvm::Type* cuda_device_ptr_ty;
    const char* cuda_module_name;
    const char* cuda_kernel_name;
};

//------------------------------------------------------------------------------

CodeGen::CodeGen(World& world, EmitHook& hook)
    : world(world)
    , hook(hook)
    , context()
    , builder(context)
    , module(new llvm::Module("anydsl", context))
    , cuda_module(new llvm::Module("a_kernel", context))
    , cuda_module_name("a_kernel.nvvm")
    , cuda_kernel_name("a_kernel")
{
    hook.assign(&builder, module);
}

// HACK -> nicer and integrated
void CodeGen::emit_cuda_decls() {
    cuda_device_ptr_ty = llvm::IntegerType::getInt64Ty(context);
    const char* thread_id_names[] = { "llvm.nvvm.read.ptx.sreg.tid.x", "llvm.nvvm.read.ptx.sreg.tid.y", "llvm.nvvm.read.ptx.sreg.tid.z" };
    const char* block_id_names[] = { "llvm.nvvm.read.ptx.sreg.ctaid.x", "llvm.nvvm.read.ptx.sreg.ctaid.y", "llvm.nvvm.read.ptx.sreg.ctaid.z" };
    const char* block_dim_names[] = { "llvm.nvvm.read.ptx.sreg.ntid.x", "llvm.nvvm.read.ptx.sreg.ntid.y", "llvm.nvvm.read.ptx.sreg.ntid.z" };
    llvm::FunctionType* thread_id_type = llvm::FunctionType::get(llvm::IntegerType::getInt32Ty(context), false);
    for (size_t i = 0; i < 3; ++i) {
        cuda_thread_id_getter[i] = llvm::Function::Create(thread_id_type, llvm::Function::ExternalLinkage, thread_id_names[i], cuda_module);
        cuda_block_id_getter[i] = llvm::Function::Create(thread_id_type, llvm::Function::ExternalLinkage, block_id_names[i], cuda_module);
        cuda_block_dim_getter[i] = llvm::Function::Create(thread_id_type, llvm::Function::ExternalLinkage, block_dim_names[i], cuda_module);
    }

    llvm::Type* void_ty = llvm::Type::getVoidTy(context);
    llvm::Type* char_ptr_ty = llvm::IntegerType::getInt8PtrTy(context);
    llvm::Type* host_data_ty = llvm::Type::getFloatPtrTy(context);
    synchronize = llvm::Function::Create(llvm::FunctionType::get(void_ty, false), llvm::Function::ExternalLinkage, "synchronize", module);
    malloc_memory = llvm::Function::Create(llvm::FunctionType::get(cuda_device_ptr_ty, { cuda_device_ptr_ty }, false), llvm::Function::ExternalLinkage, "malloc_memory", module);
    llvm::Type* write_memory_type[] = { cuda_device_ptr_ty, host_data_ty, cuda_device_ptr_ty };
    write_memory = llvm::Function::Create(llvm::FunctionType::get(void_ty, write_memory_type, false), llvm::Function::ExternalLinkage, "write_memory", module);
    llvm::Type* write_memory_type_indir[] = { cuda_device_ptr_ty, llvm::PointerType::getUnqual(host_data_ty), cuda_device_ptr_ty };
    write_memory_indir = llvm::Function::Create(llvm::FunctionType::get(void_ty, write_memory_type_indir, false), llvm::Function::ExternalLinkage, "write_memory", module);
    llvm::Type* load_kernel_type[] = { char_ptr_ty, char_ptr_ty };
    load_kernel = llvm::Function::Create(llvm::FunctionType::get(void_ty, load_kernel_type, false), llvm::Function::ExternalLinkage, "load_kernel", module);
    set_kernel_arg = llvm::Function::Create(llvm::FunctionType::get(void_ty, llvm::PointerType::getUnqual(cuda_device_ptr_ty), false), llvm::Function::ExternalLinkage, "set_kernel_arg", module);
    llvm::Type* set_problem_size_type[] = { cuda_device_ptr_ty, cuda_device_ptr_ty, cuda_device_ptr_ty };
    set_problem_size = llvm::Function::Create(llvm::FunctionType::get(void_ty, set_problem_size_type, false), llvm::Function::ExternalLinkage, "set_problem_size", module);
    launch_kernel = llvm::Function::Create(llvm::FunctionType::get(void_ty, { char_ptr_ty }, false), llvm::Function::ExternalLinkage, "launch_kernel", module);
    llvm::Type* read_memory_type[] = { cuda_device_ptr_ty, host_data_ty, cuda_device_ptr_ty };
    read_memory = llvm::Function::Create(llvm::FunctionType::get(void_ty, read_memory_type, false), llvm::Function::ExternalLinkage, "read_memory", module);
    llvm::Type* read_memory_type_indir[] = { cuda_device_ptr_ty, llvm::PointerType::getUnqual(host_data_ty), cuda_device_ptr_ty };
    read_memory_indir = llvm::Function::Create(llvm::FunctionType::get(void_ty, read_memory_type_indir, false), llvm::Function::ExternalLinkage, "read_memory", module);
    free_memory = llvm::Function::Create(llvm::FunctionType::get(void_ty, { cuda_device_ptr_ty }, false), llvm::Function::ExternalLinkage, "free_memory", module);
}

void CodeGen::emit_vector_decls() {
    const char* thread_id_name = "get_tid";
    vector_tid_type = llvm::VectorType::get(llvm::IntegerType::getInt32Ty(context), 4);
    llvm::FunctionType* thread_id_type = llvm::FunctionType::get(llvm::IntegerType::getInt32Ty(context), false);
    vector_tid_getter = llvm::Function::Create(thread_id_type, llvm::Function::ExternalLinkage, thread_id_name, module);
}
static uint64_t try_resolve_array_size(Def def) {
    // Ugly HACK
    if (const Param* p = def->isa<Param>()) {
        for (auto use : p->lambda()->uses()) {
            if (auto lambda = use->isa_lambda()) {
                if (auto larray = lambda->to()->isa_lambda()) {
                    if (larray->attribute().is(Lambda::ArrayInit)) {
                        // resolve size
                        return lambda->arg(1)->as<PrimLit>()->u64_value();
                    }
                    else if (larray->attribute().is(Lambda::StencilAr))
                        return 9;
                }
            }
        }
    }
    return 1024;
}

// HACK
void CodeGen::emit_cuda(Lambda* lambda, BBMap& bbs) {
    Lambda* target = lambda->to()->as_lambda();
    assert(target->is_builtin() && target->attribute().is(Lambda::Cuda));
    // passed lambda is the external cuda call
    const uint64_t it_space_x = try_resolve_array_size(lambda->arg(1));
    Lambda* kernel = lambda->arg(2)->as<Addr>()->lambda();
    // load kernel
    llvm::Value* module_name = builder.CreateGlobalStringPtr(cuda_module_name);
    llvm::Value* kernel_name = builder.CreateGlobalStringPtr(cuda_kernel_name);
    llvm::Value* load_args[] = { module_name, kernel_name };
    builder.CreateCall(load_kernel, load_args);
    // fetch values and create external calls for intialization
    std::vector<std::pair<llvm::Value*, llvm::Constant*>> device_ptrs;
    for (size_t i = 4, e = lambda->num_args(); i < e; ++i) {
        Def cuda_param = lambda->arg(i);
        const Type* param_type = cuda_param->type();
        uint64_t num_elems = try_resolve_array_size(cuda_param);
        llvm::Constant* size = llvm::ConstantInt::get(llvm::IntegerType::getInt64Ty(context), num_elems);
        auto alloca = builder.CreateAlloca(cuda_device_ptr_ty);
        auto device_ptr = builder.CreateCall(malloc_memory, size);
        // store device ptr
        builder.CreateStore(device_ptr, alloca);
        auto loaded_device_ptr = builder.CreateLoad(alloca);
        device_ptrs.push_back(std::pair<llvm::Value*, llvm::Constant*>(loaded_device_ptr, size));
        llvm::Value* mem_args[] = { loaded_device_ptr, lookup(cuda_param), size };
        if (param_type->isa<Ptr>() && param_type->as<Ptr>()->referenced_type()->isa<Ptr>())
            builder.CreateCall(write_memory_indir, mem_args);
        else
            builder.CreateCall(write_memory, mem_args);
        builder.CreateCall(set_kernel_arg, { alloca });
    }
    // determine problem size
    llvm::Value* problem_size_args[] = {
        llvm::ConstantInt::get(llvm::IntegerType::getInt64Ty(context), it_space_x),
        llvm::ConstantInt::get(llvm::IntegerType::getInt64Ty(context), 1),
        llvm::ConstantInt::get(llvm::IntegerType::getInt64Ty(context), 1)
    };
    builder.CreateCall(set_problem_size, problem_size_args);
    // launch
    builder.CreateCall(launch_kernel, { kernel_name });
    // synchronize
    builder.CreateCall(synchronize);

    // fetch data back to cpu
    for (size_t i = 4, e = lambda->num_args(); i < e; ++i) {
        Def cuda_param = lambda->arg(i);
        const Type* param_type = cuda_param->type();
        auto entry = device_ptrs[i-4];
        // need to fetch back memory
        llvm::Value* args[] = { entry.first, lookup(cuda_param), entry.second };
        if (param_type->isa<Ptr>() && param_type->as<Ptr>()->referenced_type()->isa<Ptr>())
            builder.CreateCall(read_memory_indir, args);
        else
            builder.CreateCall(read_memory, args);
    }

    // free memory
    for (auto device_ptr : device_ptrs)
        builder.CreateCall(free_memory, { device_ptr.first });
    // create branch to return
    builder.CreateBr(bbs[lambda->arg(3)->as_lambda()]);
}

void CodeGen::emit_vectors(llvm::Function* current, Lambda* lambda, BBMap& bbs) {
    VectorizationEntry e;
    Lambda* target = lambda->to()->as_lambda();
    assert(target->is_builtin() && target->attribute().is(Lambda::Vectorize));

    // resolve vector length
    Def vector_length = lambda->arg(1);
    assert(vector_length->isa<PrimLit>());
    e.vector_length = vector_length->as<PrimLit>()->u32_value();
    assert(e.vector_length == 4 && "FIXME");
    // loop count
    Def loop_count = lambda->arg(2);
    assert(loop_count->isa<PrimLit>());
    u32 count = loop_count->as<PrimLit>()->u32_value();
    // passed lambda is the kernel
    Lambda* kernel = lambda->arg(3)->as<Addr>()->lambda();
    Lambda* ret_lambda = lambda->arg(4)->as_lambda();
    const size_t arg_index = 5;
    const size_t num_args = lambda->num_args() >= arg_index ? lambda->num_args() - arg_index : 0;
    e.func = current;

    // build simd-function signature
    Array<llvm::Type*> simd_args(num_args);
    for (size_t i = 0; i < num_args; ++i) {
        const Type* type = lambda->arg(i + arg_index)->type();
        simd_args[i] = map(type);
    }
    llvm::FunctionType* simd_type = llvm::FunctionType::get(builder.getVoidTy(), llvm_ref(simd_args), false);
    e.kernel_simd_func = (llvm::Function*)module->getOrInsertFunction("vector_kernel_" + kernel->name, simd_type);

    // build iteration loop and wire the calls
    llvm::BasicBlock* header = llvm::BasicBlock::Create(context, "vec_header", current);
    llvm::BasicBlock* body = llvm::BasicBlock::Create(context, "vec_body", current);
    llvm::BasicBlock* exit = llvm::BasicBlock::Create(context, "vec_exit", current);
    // create loop phi and connect init value
    e.loop_counter = llvm::PHINode::Create(builder.getInt32Ty(), 2U, "vector_loop_phi", header);
    llvm::Value* i = builder.getInt32(0);
    e.loop_counter->addIncoming(i, builder.GetInsertBlock());
    // connect header
    builder.CreateBr(header);
    builder.SetInsertPoint(header);
    // create conditional branch
    llvm::Value* cond = builder.CreateICmpUGT(e.loop_counter, builder.getInt32(count));
    builder.CreateCondBr(cond, body, exit);
    // set body
    builder.SetInsertPoint(body);
    Array<llvm::Value*> args(num_args);
    for (size_t i = 0; i < num_args; ++i) {
        // check target type
        Def arg = lambda->arg(i + arg_index);
        llvm::Value* llvm_arg = lookup(arg);
        if (arg->type()->isa<Ptr>())
            llvm_arg = builder.CreateBitCast(llvm_arg, simd_args[i]);
        args[i] = llvm_arg;
    }
    // call new function
    e.kernel_func = fcts[kernel];
    e.kernel_call = builder.CreateCall(e.kernel_simd_func, llvm_ref(args));
    // inc loop counter
    e.loop_counter->addIncoming(builder.CreateAdd(e.loop_counter, builder.getInt32(e.vector_length)), body);
    builder.CreateBr(header);
    // create branch to return
    builder.SetInsertPoint(exit);
    builder.CreateBr(bbs[ret_lambda]);
    v_fcts.push_back(e);
}

void CodeGen::emit() {
    // emit cuda declarations
#if 0
    emit_cuda_decls();
#endif
    emit_vector_decls();
    std::unordered_map<Lambda*, const Param*> ret_map;
    // map all root-level lambdas to llvm function stubs
    const Param* cuda_return = 0;
    for (auto lambda : top_level_lambdas(world)) {
        if (lambda->is_builtin())
            continue;
        llvm::Function* f;
        if (lambda->is_connected_to_builtin(Lambda::Cuda)) {
            const size_t e = lambda->num_params();
            // check dimensions
            size_t i = 1;
            for (; i < 3 && lambda->param(i)->type()->isa<Pi>() && i < e; ++i)
                params[lambda->param(i)] = cuda_thread_id_getter[i - 1];
            for (; i < 5 && lambda->param(i)->type()->isa<Pi>() && i < e; ++i)
                params[lambda->param(i)] = cuda_block_id_getter[i - 3];
            for (; i < 7 && lambda->param(i)->type()->isa<Pi>() && i < e; ++i)
                params[lambda->param(i)] = cuda_block_dim_getter[i - 5];
            // cuda return param
            const Param* cuda_return = lambda->param(i);
            assert(cuda_return->type()->isa<Pi>());
            // build kernel declaration
            llvm::FunctionType* ft = llvm::cast<llvm::FunctionType>(map(world.pi(lambda->pi()->elems().slice_from_begin(i))));
            f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, cuda_kernel_name, cuda_module);
            // wire params directly
            auto arg = f->arg_begin();
            for (size_t j = i + 1; j < e; ++j) {
                llvm::Argument* param = arg++;
                const Param* p = lambda->param(j);
                param->setName(llvm::Twine(p->name));
                params[p] = param;
            }
            // append required metadata
            llvm::NamedMDNode* annotation = cuda_module->getOrInsertNamedMetadata("nvvm.annotations");
            llvm::Value* annotation_values[] = {
                f,
                llvm::MDString::get(context, "kernel"),
                llvm::ConstantInt::get(llvm::IntegerType::getInt64Ty(context), 1)
            };
            annotation->addOperand(llvm::MDNode::get(context, annotation_values));
            ret_map[lambda] = cuda_return;
        } else if (lambda->is_connected_to_builtin(Lambda::Vectorize)) {
            const size_t e = lambda->num_params();
            assert(e >= 2 && "at least a thread id and a return expected");
            // check dimensions
            const Param* tid = lambda->param(1);
            params[tid] = vector_tid_getter;
            const Param* vector_return = lambda->param(2);
            assert(vector_return->type()->isa<Pi>());
            // build vector declaration
            llvm::FunctionType* ft = llvm::cast<llvm::FunctionType>(map(world.pi(lambda->pi()->elems().slice_from_begin(2))));
            f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, lambda->name, module);
            // wire params directly
            auto arg = f->arg_begin();
            for (size_t i = 3; i < e; ++i) {
                llvm::Argument* param = arg++;
                const Param* p = lambda->param(i);
                param->setName(llvm::Twine(p->name));
                params[p] = param;
            }
            ret_map[lambda] = vector_return;
        } else {
            llvm::FunctionType* ft = llvm::cast<llvm::FunctionType>(map(lambda->type()));
            f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, lambda->name, module);
        }
        fcts.emplace(lambda, f);
    }

    // for all top-level functions
    for (auto lf : fcts) {
        Lambda* lambda = lf.first;
        if (lambda->is_builtin() || lambda->empty())
            continue;

        assert(lambda->is_returning() || lambda->is_connected_to_builtin());
        llvm::Function* fct = lf.second;

        // map params
        const Param* ret_param = 0;
        if (lambda->is_connected_to_builtin())
            ret_param = ret_map[lambda];
        else {
            auto arg = fct->arg_begin();
            for (auto param : lambda->params()) {
                if (param->type()->isa<Mem>())
                    continue;
                if (param->order() == 0) {
                    arg->setName(param->name);
                    params[param] = arg++;
                }
                else {
                    assert(!ret_param);
                    ret_param = param;
                }
            }
        }
        assert(ret_param);

        Scope scope(lambda);
        BBMap bbs;

        for (auto lambda : scope.rpo()) {
            // map all bb-like lambdas to llvm bb stubs
            auto bb = bbs[lambda] = llvm::BasicBlock::Create(context, lambda->name, fct);

            // create phi node stubs (for all non-cascading lambdas different from entry)
            if (!lambda->is_cascading() && !scope.is_entry(lambda)) {
                for (auto param : lambda->params())
                    if (!param->type()->isa<Mem>())
                        phis[param] = llvm::PHINode::Create(map(param->type()), (unsigned) param->peek().size(), param->name, bb);
            }

        }

        Schedule schedule = schedule_smart(scope);

        // emit body for each bb
        for (auto lambda : scope.rpo()) {
            if (lambda->empty())
                continue;
            assert(scope.is_entry(lambda) || lambda->is_basicblock());
            builder.SetInsertPoint(bbs[lambda]);

            for (auto primop :  schedule[lambda]) {
                // skip higher-order primops, stuff dealing with frames and all memory related stuff except stores
                if (!primop->type()->isa<Pi>() && !primop->type()->isa<Frame>()
                        && (!primop->type()->isa<Mem>() || primop->isa<Store>()))
                    primops[primop] = emit(primop);
            }

            // terminate bb
            if (lambda->to() == ret_param) { // return
                size_t num_args = lambda->num_args();
                switch (num_args) {
                    case 0: builder.CreateRetVoid(); break;
                    case 1:
                        if (lambda->arg(0)->type()->isa<Mem>())
                            builder.CreateRetVoid();
                        else
                            builder.CreateRet(lookup(lambda->arg(0)));
                        break;
                    case 2: {
                        if (lambda->arg(0)->type()->isa<Mem>()) {
                            builder.CreateRet(lookup(lambda->arg(1)));
                            break;
                        } else if (lambda->arg(1)->type()->isa<Mem>()) {
                            builder.CreateRet(lookup(lambda->arg(0)));
                            break;
                        }
                        // FALLTHROUGH
                    }
                    default: {
                        Array<llvm::Value*> values(num_args);
                        Array<llvm::Type*> elems(num_args);

                        size_t n = 0;
                        for (size_t a = 0; a < num_args; ++a) {
                            if (!lambda->arg(n)->type()->isa<Mem>()) {
                                llvm::Value* val = lookup(lambda->arg(a));
                                values[n] = val;
                                elems[n++] = val->getType();
                            }
                        }

                        assert(n == num_args || n+1 == num_args);
                        values.shrink(n);
                        elems.shrink(n);
                        llvm::Value* agg = llvm::UndefValue::get(llvm::StructType::get(context, llvm_ref(elems)));

                        for (size_t i = 0; i != n; ++i)
                            agg = builder.CreateInsertValue(agg, values[i], { unsigned(i) });

                        builder.CreateRet(agg);
                        break;
                    }
                }
            } else if (auto select = lambda->to()->isa<Select>()) { // conditional branch
                llvm::Value* cond = lookup(select->cond());
                llvm::BasicBlock* tbb = bbs[select->tval()->as_lambda()];
                llvm::BasicBlock* fbb = bbs[select->fval()->as_lambda()];
                builder.CreateCondBr(cond, tbb, fbb);
            } else {
                if (auto higher_order_call = lambda->to()->isa<Param>()) { // higher-order call
                    llvm::CallInst* call_target = builder.CreateCall(params[higher_order_call]);
                    auto succ = lambda->arg(1)->as_lambda();
                    const Param* param = succ->param(0)->type()->isa<Mem>() ? nullptr : succ->param(0);
                    if (param == nullptr && succ->num_params() == 2)
                        param = succ->param(1);
                    params[param] = call_target;
                    builder.CreateBr(bbs[succ]);
                } else {
                    Lambda* to_lambda = lambda->to()->as_lambda();
                    if (to_lambda->is_basicblock())      // ordinary jump
                        builder.CreateBr(bbs[to_lambda]);
                    else {
                        if (lambda->to()->isa<Lambda>()) {
                            const Lambda::Attribute& attr = lambda->to()->as_lambda()->attribute();
                            if (attr.is(Lambda::Cuda))
                                emit_cuda(lambda, bbs);
                            else if(attr.is(Lambda::Vectorize))
                                emit_vectors(fct, lambda, bbs);
                            else
                                goto no_lambda;
                        } else {
no_lambda:
                            // put all first-order args into an array
                            Array<llvm::Value*> args(lambda->args().size() - 1);
                            size_t i = 0;
                            Def ret_arg = 0;
                            for (auto arg : lambda->args()) {
                                if (arg->order() == 0) {
                                    if (!arg->type()->isa<Mem>())
                                        args[i++] = lookup(arg);
                                }
                                else {
                                    assert(!ret_arg);
                                    ret_arg = arg;
                                }
                            }
                            args.shrink(i);
                            llvm::CallInst* call = builder.CreateCall(fcts[to_lambda], llvm_ref(args));

                            if (ret_arg == ret_param)       // call + return
                                builder.CreateRet(call);
                            else {                          // call + continuation
                                Lambda* succ = ret_arg->as_lambda();
                                const Param* param = succ->param(0)->type()->isa<Mem>() ? nullptr : succ->param(0);
                                if (param == nullptr && succ->num_params() == 2)
                                    param = succ->param(1);

                                builder.CreateBr(bbs[succ]);
                                if (param) {
                                    auto i = phis.find(param);
                                    if (i != phis.end())
                                        i->second->addIncoming(call, builder.GetInsertBlock());
                                    else
                                        params[param] = call;
                                }
                            }
                        }
                    }
                }
            }
        }

        // add missing arguments to phis
        for (auto p : phis) {
            const Param* param = p.first;
            llvm::PHINode* phi = p.second;

            for (auto peek : param->peek())
                phi->addIncoming(lookup(peek.def()), bbs[peek.from()]);
        }

        // FIXME: params.clear();
        phis.clear();
        primops.clear();
    }

#ifndef NDEBUG
    llvm::verifyModule(*this->module);
    llvm::verifyModule(*this->cuda_module);
#endif
}

void CodeGen::postprocess() {
#ifdef WFV2_SUPPORT
    if (v_fcts.size() < 1)
        return;
    // vectorize entries
    for (auto& entry : v_fcts) {
       WFVInterface::WFVInterface wfv(module, &context, entry.kernel_func, entry.kernel_simd_func, entry.vector_length);
       bool b_simd = wfv.addSIMDSemantics(*vector_tid_getter, false, true, false, false, false, true, false, true, false, true);
       assert(b_simd && "simd semantics for vectorization failed");
       bool b = wfv.run();
       assert(b && "vectorization failed");
       // inline kernel
       llvm::InlineFunctionInfo info;
       llvm::InlineFunction(entry.kernel_call, info);

       std::vector<llvm::CallInst*> calls;
       for (auto it = vector_tid_getter->use_begin(), e = vector_tid_getter->use_end(); it != e; ++it) {
           if (auto call = llvm::dyn_cast<llvm::CallInst>(*it))
               if (const Function* func = call->getParent()->getParent())
                   if (func == entry.func)
                       calls.push_back(call);
       }
       for (auto it = calls.rbegin(), e = calls.rend(); it != e; ++it) {
           BasicBlock::iterator ii(*it);
           ReplaceInstWithValue((*it)->getParent()->getInstList(), ii, entry.loop_counter);
       }
    }
#endif
}

void CodeGen::dump() {
    module->dump();
    cuda_module->dump();
}

llvm::Value* CodeGen::lookup(Def def) {
    if (def->is_const())
        return emit(def);

    if (auto primop = def->isa<PrimOp>())
        return primops[primop];

    const Param* param = def->as<Param>();
    auto i = params.find(param);
    if (i != params.end())
        return i->second;

    assert(phis.find(param) != phis.end());
    return phis[param];
}

llvm::Value* CodeGen::emit(Def def) {
    if (auto bin = def->isa<BinOp>()) {
        llvm::Value* lhs = lookup(bin->lhs());
        llvm::Value* rhs = lookup(bin->rhs());

        if (auto rel = def->isa<RelOp>()) {
            switch (rel->relop_kind()) {
                case RelOp_cmp_eq:   return builder.CreateICmpEQ (lhs, rhs);
                case RelOp_cmp_ne:   return builder.CreateICmpNE (lhs, rhs);

                case RelOp_cmp_ugt:  return builder.CreateICmpUGT(lhs, rhs);
                case RelOp_cmp_uge:  return builder.CreateICmpUGE(lhs, rhs);
                case RelOp_cmp_ult:  return builder.CreateICmpULT(lhs, rhs);
                case RelOp_cmp_ule:  return builder.CreateICmpULE(lhs, rhs);

                case RelOp_cmp_sgt:  return builder.CreateICmpSGT(lhs, rhs);
                case RelOp_cmp_sge:  return builder.CreateICmpSGE(lhs, rhs);
                case RelOp_cmp_slt:  return builder.CreateICmpSLT(lhs, rhs);
                case RelOp_cmp_sle:  return builder.CreateICmpSLE(lhs, rhs);

                case RelOp_fcmp_oeq: return builder.CreateFCmpOEQ(lhs, rhs);
                case RelOp_fcmp_one: return builder.CreateFCmpONE(lhs, rhs);

                case RelOp_fcmp_ogt: return builder.CreateFCmpOGT(lhs, rhs);
                case RelOp_fcmp_oge: return builder.CreateFCmpOGE(lhs, rhs);
                case RelOp_fcmp_olt: return builder.CreateFCmpOLT(lhs, rhs);
                case RelOp_fcmp_ole: return builder.CreateFCmpOLE(lhs, rhs);

                case RelOp_fcmp_ueq: return builder.CreateFCmpUEQ(lhs, rhs);
                case RelOp_fcmp_une: return builder.CreateFCmpUNE(lhs, rhs);

                case RelOp_fcmp_ugt: return builder.CreateFCmpUGT(lhs, rhs);
                case RelOp_fcmp_uge: return builder.CreateFCmpUGE(lhs, rhs);
                case RelOp_fcmp_ult: return builder.CreateFCmpULT(lhs, rhs);
                case RelOp_fcmp_ule: return builder.CreateFCmpULE(lhs, rhs);

                case RelOp_fcmp_uno: return builder.CreateFCmpUNO(lhs, rhs);
                case RelOp_fcmp_ord: return builder.CreateFCmpORD(lhs, rhs);
            }
        }

        switch (def->as<ArithOp>()->arithop_kind()) {
            case ArithOp_add:  return builder.CreateAdd (lhs, rhs);
            case ArithOp_sub:  return builder.CreateSub (lhs, rhs);
            case ArithOp_mul:  return builder.CreateMul (lhs, rhs);
            case ArithOp_udiv: return builder.CreateUDiv(lhs, rhs);
            case ArithOp_sdiv: return builder.CreateSDiv(lhs, rhs);
            case ArithOp_urem: return builder.CreateURem(lhs, rhs);
            case ArithOp_srem: return builder.CreateSRem(lhs, rhs);

            case ArithOp_fadd: return builder.CreateFAdd(lhs, rhs);
            case ArithOp_fsub: return builder.CreateFSub(lhs, rhs);
            case ArithOp_fmul: return builder.CreateFMul(lhs, rhs);
            case ArithOp_fdiv: return builder.CreateFDiv(lhs, rhs);
            case ArithOp_frem: return builder.CreateFRem(lhs, rhs);

            case ArithOp_and:  return builder.CreateAnd (lhs, rhs);
            case ArithOp_or:   return builder.CreateOr  (lhs, rhs);
            case ArithOp_xor:  return builder.CreateXor (lhs, rhs);

            case ArithOp_shl:  return builder.CreateShl (lhs, rhs);
            case ArithOp_lshr: return builder.CreateLShr(lhs, rhs);
            case ArithOp_ashr: return builder.CreateAShr(lhs, rhs);
        }
    }

    if (auto conv = def->isa<ConvOp>()) {
        llvm::Value* from = lookup(conv->from());
        llvm::Type* to = map(conv->type());

        switch (conv->convop_kind()) {
            case ConvOp_trunc:    return builder.CreateTrunc   (from, to);
            case ConvOp_zext:     return builder.CreateZExt    (from, to);
            case ConvOp_sext:     return builder.CreateSExt    (from, to);
            case ConvOp_stof:     return builder.CreateSIToFP  (from, to);
            case ConvOp_utof:     return builder.CreateSIToFP  (from, to);
            case ConvOp_ftrunc:   return builder.CreateFPTrunc (from, to);
            case ConvOp_ftos:     return builder.CreateFPToSI  (from, to);
            case ConvOp_ftou:     return builder.CreateFPToUI  (from, to);
            case ConvOp_fext:     return builder.CreateFPExt   (from, to);
            case ConvOp_bitcast:  return builder.CreateBitCast (from, to);
            case ConvOp_inttoptr: return builder.CreateIntToPtr(from, to);
            case ConvOp_ptrtoint: return builder.CreatePtrToInt(from, to);
        }
    }

    if (auto select = def->isa<Select>()) {
        llvm::Value* cond = lookup(select->cond());
        llvm::Value* tval = lookup(select->tval());
        llvm::Value* fval = lookup(select->fval());
        return builder.CreateSelect(cond, tval, fval);
    }

#if 0
    if (auto array = def->isa<ArrayValue>()) {
        auto alloca = new llvm::AllocaInst(
            llvm::ArrayType::get(map(array->array_type()->elem_type()), array->size()),     // type
            nullptr /* no variable length array */, array->name,
            builder.GetInsertBlock()->getParent()->getEntryBlock().getTerminator()          // insert before this
        );

        llvm::Instruction* before = builder.GetInsertBlock()->getParent()->getEntryBlock().getTerminator();
        u64 i = 0;
        for (auto op : array->ops()) {
            auto gep = llvm::GetElementPtrInst::CreateInBounds(
                alloca, 
                { llvm::ConstantInt::get(llvm::IntegerType::getInt64Ty(context), i) },
                op->name, before);
            before = new llvm::StoreInst(lookup(op), gep, before);
        }
        return alloca;
    }

    if (auto arrayop = def->isa<ArrayOp>()) {
        auto array = lookup(arrayop->array());
        auto gep = builder.CreateInBoundsGEP(array, lookup(arrayop->index()));
        if (auto extract = arrayop->isa<ArrayExtract>())
            return builder.CreateLoad(gep, extract->name);
        return builder.CreateStore(lookup(arrayop->as<ArrayInsert>()->value()), gep);
    }
#endif

    if (auto tuple = def->isa<Tuple>()) {
        llvm::Value* agg = llvm::UndefValue::get(map(tuple->type()));
        for (size_t i = 0, e = tuple->ops().size(); i != e; ++i)
            agg = builder.CreateInsertValue(agg, lookup(tuple->op(i)), { unsigned(i) });
        return agg;
    }

    if (auto aggop = def->isa<AggOp>()) {
        auto tuple = lookup(aggop->agg());
        unsigned idx = aggop->index()->primlit_value<unsigned>();

        if (auto extract = aggop->as<Extract>()) {
            if (extract->agg()->isa<Load>())
                return tuple; // bypass artificial extract
            return builder.CreateExtractValue(tuple, { idx });
        }

        auto insert = def->as<Insert>();
        auto value = lookup(insert->value());

        return builder.CreateInsertValue(tuple, value, { idx });
    }

    if (auto primlit = def->isa<PrimLit>()) {
        llvm::Type* type = map(primlit->type());
        Box box = primlit->value();

        switch (primlit->primtype_kind()) {
            case PrimType_u1:  return builder.getInt1(box.get_u1().get());
            case PrimType_u8:  return builder.getInt8(box.get_u8());
            case PrimType_u16: return builder.getInt16(box.get_u16());
            case PrimType_u32: return builder.getInt32(box.get_u32());
            case PrimType_u64: return builder.getInt64(box.get_u64());
            case PrimType_f32: return llvm::ConstantFP::get(type, box.get_f32());
            case PrimType_f64: return llvm::ConstantFP::get(type, box.get_f64());
        }
    }

    if (auto undef = def->isa<Undef>()) // bottom and any
        return llvm::UndefValue::get(map(undef->type()));

    if (auto load = def->isa<Load>())
        return builder.CreateLoad(lookup(load->ptr()));

    if (auto store = def->isa<Store>())
        return builder.CreateStore(lookup(store->val()), lookup(store->ptr()));

    if (auto slot = def->isa<Slot>())
        return builder.CreateAlloca(map(slot->type()->as<Ptr>()->referenced_type()), 0, slot->unique_name());

    if (def->isa<Enter>() || def->isa<Leave>())
        return 0;

    if (auto vector = def->isa<Vector>()) {
        llvm::Value* vec = llvm::UndefValue::get(map(vector->type()));
        for (size_t i = 0, e = vector->size(); i != e; ++i)
            vec = builder.CreateInsertElement(vec, lookup(vector->op(i)), lookup(world.literal_u32(i)));

        return vec;
    }

    if (auto lea = def->isa<LEA>()) {
        if (lea->referenced_type()->isa<Sigma>())
            return builder.CreateConstInBoundsGEP2_64(lookup(lea->ptr()), 0ull, lea->index()->primlit_value<u64>());

        assert(lea->referenced_type()->isa<ArrayType>());
        return builder.CreateInBoundsGEP(lookup(lea->ptr()), lookup(lea->index()));
    }

    if (auto addr = def->isa<Addr>())
        return fcts[addr->lambda()];

    assert(!def->is_corenode());
    return hook.emit(def);
}

llvm::Type* CodeGen::map(const Type* type) {
    assert(!type->isa<Mem>());
    llvm::Type* llvm_type;
    switch (type->kind()) {
        case Node_PrimType_u1:  llvm_type = llvm::IntegerType::get(context,  1); break;
        case Node_PrimType_u8:  llvm_type = llvm::IntegerType::get(context,  8); break;
        case Node_PrimType_u16: llvm_type = llvm::IntegerType::get(context, 16); break;
        case Node_PrimType_u32: llvm_type = llvm::IntegerType::get(context, 32); break;
        case Node_PrimType_u64: llvm_type = llvm::IntegerType::get(context, 64); break;
        case Node_PrimType_f32: llvm_type = llvm::Type::getFloatTy(context);     break;
        case Node_PrimType_f64: llvm_type = llvm::Type::getDoubleTy(context);    break;
        case Node_Ptr:          llvm_type = llvm::PointerType::getUnqual(map(type->as<Ptr>()->referenced_type())); break;
        case Node_ArrayType:    return map(type->as<ArrayType>()->elem_type());
        case Node_Pi: {
            // extract "return" type, collect all other types
            const Pi* pi = type->as<Pi>();
            llvm::Type* ret = 0;
            size_t i = 0;
            Array<llvm::Type*> elems(pi->size() - 1);
            for (auto elem : pi->elems()) {
                if (elem->isa<Mem>())
                    continue;
                if (auto pi = elem->isa<Pi>()) {
                    assert(!ret && "only one 'return' supported");
                    if (pi->empty())
                        ret = llvm::Type::getVoidTy(context);
                    else if (pi->size() == 1)
                        ret = pi->elem(0)->isa<Mem>() ? llvm::Type::getVoidTy(context) : map(pi->elem(0));
                    else if (pi->size() == 2) {
                        if (pi->elem(0)->isa<Mem>())
                            ret = map(pi->elem(1));
                        else if (pi->elem(1)->isa<Mem>())
                            ret = map(pi->elem(0));
                        else
                            goto multiple;
                    } else {
multiple:
                        Array<llvm::Type*> elems(pi->size());
                        size_t num = 0;
                        for (size_t j = 0, e = elems.size(); j != e; ++j) {
                            if (pi->elem(j)->isa<Mem>())
                                continue;
                            ++num;
                            elems[j] = map(pi->elem(j));
                        }
                        elems.shrink(num);
                        ret = llvm::StructType::get(context, llvm_ref(elems));
                    }
                } else
                    elems[i++] = map(elem);
            }
            elems.shrink(i);
            assert(ret);

            return llvm::FunctionType::get(ret, llvm_ref(elems), false);
        }

        case Node_Sigma: {
            // TODO watch out for cycles!
            const Sigma* sigma = type->as<Sigma>();
            Array<llvm::Type*> elems(sigma->size());
            size_t num = 0;
            for (auto elem : sigma->elems())
                elems[num++] = map(elem);
            elems.shrink(num);
            return llvm::StructType::get(context, llvm_ref(elems));
        }

        default:
            assert(!type->is_corenode());
            return hook.map(type);
    }

    if (type->length() == 1)
        return llvm_type;
    return llvm::VectorType::get(llvm_type, type->length());
}

//------------------------------------------------------------------------------

void emit_llvm(World& world, EmitHook& hook) {
    CodeGen cg(world, hook);
    cg.emit();
    cg.postprocess();
    cg.dump();
}

//------------------------------------------------------------------------------

} // namespace thorin

#endif
