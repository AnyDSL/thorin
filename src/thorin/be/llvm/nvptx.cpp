#include "thorin/be/llvm/nvptx.h"

#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>

namespace thorin {

NVPTXCodeGen::NVPTXCodeGen(World& world, const Cont2Config& kernel_config) : NVVMCodeGen(world, kernel_config) {
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
}

void NVPTXCodeGen::emit(std::ostream& stream, int opt, bool debug) {
    auto& llvm_module = CodeGen::emit(opt, debug);

    auto triple_str = llvm_module->getTargetTriple();
    std::string error;
    auto target = llvm::TargetRegistry::lookupTarget(triple_str, error);
    std::string cpu("sm_20");
    std::string attrs("+ptx32");
    llvm::TargetOptions options;
    options.AllowFPOpFusion = llvm::FPOpFusion::Fast;
    std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(triple_str, cpu, attrs, options, llvm::Reloc::PIC_, llvm::CodeModel::Default, llvm::CodeGenOpt::Aggressive));

    llvm_module->addModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz", 1);
    for (auto &fun : *llvm_module)
        fun.addFnAttr("nvptx-f32ftz", "true");
    llvm::legacy::FunctionPassManager function_pass_manager(llvm_module.get());
    llvm::legacy::PassManager module_pass_manager;

    module_pass_manager.add(llvm::createTargetTransformInfoWrapperPass(machine->getTargetIRAnalysis()));
    function_pass_manager.add(llvm::createTargetTransformInfoWrapperPass(machine->getTargetIRAnalysis()));

    llvm::PassManagerBuilder builder;
    builder.OptLevel = opt;
    builder.Inliner = llvm::createFunctionInliningPass(builder.OptLevel, 0, false);
    machine->adjustPassManager(builder);
    builder.populateFunctionPassManager(function_pass_manager);
    builder.populateModulePassManager(module_pass_manager);

    machine->Options.MCOptions.AsmVerbose = true;

    llvm::SmallString<0> outstr;
    llvm::raw_svector_ostream llvm_stream(outstr);

    machine->addPassesToEmitFile(module_pass_manager, llvm_stream, llvm::TargetMachine::CGFT_AssemblyFile, true);

    function_pass_manager.doInitialization();
    for (auto func = llvm_module->begin(); func != llvm_module->end(); ++func)
        function_pass_manager.run(*func);
    function_pass_manager.doFinalization();
    module_pass_manager.run(*llvm_module);

    stream << outstr.c_str();
}

}
