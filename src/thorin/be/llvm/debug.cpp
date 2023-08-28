#include "llvm.h"

#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/DebugInfo/DWARF/DWARFTypeUnit.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Host.h>

namespace thorin::llvm {

CodeGen::Debug::Debug(thorin::llvm::CodeGen& cg) : cg_(cg), dibuilder_(cg.module()) {}

void CodeGen::Debug::emit_module() {
    cg_.module().addModuleFlag(llvm::Module::Warning, "Debug Info Version", llvm::DEBUG_METADATA_VERSION);
    // Darwin only supports dwarf2
    if (llvm::Triple(llvm::sys::getProcessTriple()).isOSDarwin())
        cg_.module().addModuleFlag(llvm::Module::Warning, "Dwarf Version", 2);
    dicompile_unit_ = dibuilder_.createCompileUnit(llvm::dwarf::DW_LANG_C, dibuilder_.createFile(cg_.world().name(), llvm::StringRef()), "Impala", cg_.opt() > 0, llvm::StringRef(), 0);
}

void CodeGen::Debug::prepare(const thorin::Scope& scope, llvm::Function* fct) {
    auto difile = get_difile(cg_.entry_->loc().file);
    auto disub_program = dibuilder_.createFunction(
            discope_, fct->getName(), fct->getName(), difile, cg_.entry_->loc().begin.row,
            dibuilder_.createSubroutineType(dibuilder_.getOrCreateTypeArray(llvm::ArrayRef<llvm::Metadata*>())),
            cg_.entry_->loc().begin.row,
            llvm::DINode::FlagPrototyped,
            llvm::DISubprogram::SPFlagDefinition | (cg_.opt() > 0 ? llvm::DISubprogram::SPFlagOptimized : llvm::DISubprogram::SPFlagZero));
    fct->setSubprogram(disub_program);
    discope_ = disub_program;
}

void CodeGen::Debug::prepare(llvm::IRBuilder<>& irbuilder, const thorin::Continuation* cont) {
    irbuilder.SetCurrentDebugLocation(llvm::DILocation::get(discope_->getContext(), cont->loc().begin.row, cont->loc().begin.col, discope_));
}

void CodeGen::Debug::prepare(llvm::IRBuilder<>& irbuilder, const thorin::Def* def) {
    irbuilder.SetCurrentDebugLocation(llvm::DILocation::get(discope_->getContext(), def->loc().begin.row, def->loc().begin.col, discope_));
}

void CodeGen::Debug::finalize(const thorin::Def* def, llvm::Value* val) {
    val->setName(def->unique_name());
}

void CodeGen::Debug::register_param(const Param* param, int index, llvm::Value* val) {
    if (!param->type()->isa<PrimType>())
        return;
    llvm::DIFile* file = get_difile(param->debug().loc.file);
    auto local_var = dibuilder_.createParameterVariable(discope_, param->name(), index, file, param->debug().loc.begin.row, get_ditype(param->type()));
    dibuilder_.insertDbgValueIntrinsic(val, local_var, llvm::DIExpression::get(cg_.context(), {}), get_dilocation(param->debug().loc, discope_), cg_.cont2bb(param->continuation()));
}

llvm::DIFile* CodeGen::Debug::get_difile(const std::string& file) {
    auto src_file = llvm::sys::path::filename(file);
    auto src_dir = llvm::sys::path::parent_path(file);
    return dibuilder_.createFile(src_file, src_dir);
}

llvm::DILocation* CodeGen::Debug::get_dilocation(const thorin::Loc& loc, llvm::DIScope* discope) {
    return llvm::DILocation::get(cg_.context(), loc.begin.row, loc.begin.col, discope);
}

llvm::DIType* CodeGen::Debug::get_ditype(const thorin::Type* t) {
    if (auto found = types_.lookup(t))
        return *found;

    using namespace llvm::dwarf;

    if (auto prim_type = t->isa<PrimType>()) {
        switch (prim_type->primtype_tag()) {
            case PrimType_bool:                                                             return types_[t] = dibuilder_.createBasicType("bool", 1, DW_ATE_boolean);
            case PrimType_ps8:  case PrimType_qs8:  case PrimType_pu8:  case PrimType_qu8:  return types_[t] = dibuilder_.createBasicType("bool", 8, is_type_s(prim_type) ? DW_ATE_signed_char : DW_ATE_unsigned_char);
            case PrimType_ps16: case PrimType_qs16: case PrimType_pu16: case PrimType_qu16: return types_[t] = dibuilder_.createBasicType("bool", 16, is_type_s(prim_type) ? DW_ATE_signed : DW_ATE_unsigned);
            case PrimType_ps32: case PrimType_qs32: case PrimType_pu32: case PrimType_qu32: return types_[t] = dibuilder_.createBasicType("bool", 32, is_type_s(prim_type) ? DW_ATE_signed : DW_ATE_unsigned);
            case PrimType_ps64: case PrimType_qs64: case PrimType_pu64: case PrimType_qu64: return types_[t] = dibuilder_.createBasicType("bool", 64, is_type_s(prim_type) ? DW_ATE_signed : DW_ATE_unsigned);
            case PrimType_pf16: case PrimType_qf16:                                         return types_[t] = dibuilder_.createBasicType("bool", 16, DW_ATE_float);
            case PrimType_pf32: case PrimType_qf32:                                         return types_[t] = dibuilder_.createBasicType("bool", 32, DW_ATE_float);
            case PrimType_pf64: case PrimType_qf64:                                         return types_[t] = dibuilder_.createBasicType("bool", 64, DW_ATE_float);
        }
    } else if (auto ptr_t = t->isa<PtrType>()) {
        return types_[t] = dibuilder_.createPointerType(get_ditype(ptr_t->pointee()), cg_.machine_->getPointerSize(static_cast<unsigned int>(ptr_t->addr_space())));
    } else if (auto fun_t = t->isa<FnType>()) {

    } else if (auto agg_t = t->isa<StructType>()) {
        types_[t] = dibuilder_.createForwardDecl(DW_TAG_structure_type, agg_t->name().str(), dicompile_unit_, get_difile(t->debug().loc.file), t->debug().loc.begin.row);
        auto data_layout = cg_.module().getDataLayout();
        auto layout_t = data_layout.getStructLayout(llvm::cast<llvm::StructType>(cg_.convert(t)));
        std::vector<llvm::Metadata*> members;
        size_t i = 0;
        for (auto member_t : agg_t->types()) {
            auto member_di = get_ditype(member_t);
            auto member_llvm_t = cg_.convert(member_t);
            auto derived = dibuilder_.createMemberType(dicompile_unit_, agg_t->op_name(i).str(), get_difile(t->debug().loc.file), t->debug().loc.begin.row,
                                                       data_layout.getTypeSizeInBits(member_llvm_t),
                                                       data_layout.getABITypeAlign(member_llvm_t).value(),
                                                       layout_t->getElementOffsetInBits(i),
                                                       llvm::DINode::DIFlags::FlagZero,
                                                       member_di);
            i++;
        }
        return types_[t] = dibuilder_.createStructType(dicompile_unit_, agg_t->name().str(), get_difile(t->debug().loc.file), t->debug().loc.begin.row,
                                                       layout_t->getSizeInBits(),
                                                       layout_t->getAlignment().value(),
                                                       llvm::DINode::DIFlags::FlagZero,
                                                       nullptr,
                                                       dibuilder_.getOrCreateArray(members));
    }

    assert(false);
}

void CodeGen::Debug::finalize() {
    dibuilder_.finalize();
}

}