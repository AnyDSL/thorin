#include "json.h"

namespace thorin::json {

class TypeTable {
public:
    json nominal_fwd_table = json::array();
    json type_table = json::array();

    DefMap<std::string> known_types;

    std::string translate_type (const Def* def) {
        const Type * type = def->as<Type>();
        auto it = known_types.find(type);
        if (it != known_types.end()) {
            return it->second;
        }

        json result;
        if (auto arr = type->isa<DefiniteArrayType>()) {
            auto elem_type = translate_type(arr->elem_type());

            result["type"] = "def_array";
            result["args"] = { elem_type };
            result["length"] = arr->dim();
            result["name"] = elem_type + "_darr_" + std::to_string(type_table.size());
        } else if (auto arr = type->isa<IndefiniteArrayType>()) {
            auto elem_type = translate_type(arr->elem_type());

            result["type"] = "indef_array";
            result["args"] = { elem_type };
            result["name"] = elem_type + "_iarr_" + std::to_string(type_table.size());
        } else if (type->isa<BottomType>()) {
            result["name"] = "bottom_t";
            result["type"] = "bottom";
        } else if (auto fntype = type->isa<FnType>()) {
            json arg_types = json::array();
            for (auto arg : fntype->ops()) {
                arg_types.push_back(translate_type(arg));
            }

            result["type"] = "function";
            result["name"] = "_" + std::to_string(type_table.size());
            result["args"] = arg_types;
        } else if (auto closuretype = type->isa<ClosureType>()) {
            json args = json::array();
            for (auto arg : closuretype->ops()) {
                args.push_back(translate_type(arg));
            }

            result["type"] = "closure";
            result["name"] = "_" + std::to_string(type_table.size());
            result["args"] = args;
        } else if (type->isa<FrameType>()) {
            result["name"] = "frame_t";
            result["type"] = "frame";
        } else if (type->isa<MemType>()) {
            result["name"] = "mem_t";
            result["type"] = "mem";
        } else if (auto structtype = type->isa<StructType>()) {
            auto name = "_struct_" + std::to_string(nominal_fwd_table.size());
            known_types[type] = name;

            json arg_names = json::array();
            for (size_t i = 0; i < structtype->num_ops(); ++i) {
                arg_names.push_back(structtype->op_name(i).str());
            }

            json forward_decl;
            forward_decl["name"] = name;
            forward_decl["type"] = "struct";
            forward_decl["struct_name"] = structtype->name().str();
            forward_decl["arg_names"] = arg_names;
            nominal_fwd_table.push_back(forward_decl);

            json args = json::array();
            for (size_t i = 0; i < structtype->num_ops(); ++i) {
                args.push_back(translate_type(structtype->op(i)));
            }

            result["type"] = "struct";
            result["name"] = name;
            result["struct_name"] = structtype->name().str();
            result["arg_names"] = arg_names;
            result["args"] = args;
        } else if (auto varianttype = type->isa<VariantType>()) {
            auto name = "_variant_" + std::to_string(nominal_fwd_table.size());
            known_types[type] = name;

            json arg_names = json::array();
            for (size_t i = 0; i < varianttype->num_ops(); ++i) {
                arg_names.push_back(varianttype->op_name(i).str());
            }

            json forward_decl;
            forward_decl["name"] = name;
            forward_decl["type"] = "variant";
            forward_decl["variant_name"] = varianttype->name().str();
            forward_decl["arg_names"] = arg_names;
            nominal_fwd_table.push_back(forward_decl);

            json args = json::array();
            for (size_t i = 0; i < varianttype->num_ops(); ++i) {
                args.push_back(translate_type(varianttype->op(i)));
            }

            result["type"] = "variant";
            result["name"] = name;
            result["variant_name"] = varianttype->name().str();
            result["args"] = args;
            result["arg_names"] = arg_names;
        } else if (auto tupletype = type->isa<TupleType>()) {
            json args = json::array();
            for (size_t i = 0; i < tupletype->num_ops(); ++i) {
                args.push_back(translate_type(tupletype->op(i)));
            }

            result["type"] = "tuple";
            result["name"] = "_" + std::to_string(type_table.size());
            result["args"] = args;
        } else if (auto prim = type->isa<PrimType>()) {
            result["name"] = "_" + std::to_string(type_table.size());
            result["type"] = "prim";
            switch (prim->primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case PrimTypeTag::PrimType_##T: { result["tag"] = #T; break; }
#include <thorin/tables/primtypetable.h>
            }
        } else if (auto ptrtype = type->isa<PtrType>()) {
            auto pointee_type = translate_type(ptrtype->pointee());

            result["type"] = "ptr";
            result["args"] = { pointee_type };
            result["name"] = pointee_type + "_p_" + std::to_string(type_table.size());
            switch (ptrtype->addr_space()) {
            case AddrSpace::Generic:
                //result["addrspace"] = "generic"; //Default
                break;
            case AddrSpace::Global:
                result["addrspace"] = "global";
                break;
            case AddrSpace::Texture:
                result["addrspace"] = "texture";
                break;
            case AddrSpace::Shared:
                result["addrspace"] = "shared";
                break;
            case AddrSpace::Constant:
                result["addrspace"] = "constant";
                break;
            case AddrSpace::Private:
                result["addrspace"] = "private";
                break;
            }
        } else if (auto vectype = type->isa<VectorType>()) {
            result["type"] = "ptr";
            result["scalar"] = translate_type(vectype->scalarize());
            result["length"] = vectype->length();
        } else {
            std::cerr << "type cannot be translated\n";
            type->dump();
            THORIN_UNREACHABLE;
        }
        known_types[type] = result["name"];
        type_table.push_back(result);
        return result["name"];
    }
};

class DefTable {
public:
    DefTable(TypeTable& type_table) : type_table_(type_table) {}

    json decl_table = json::array();
    json def_table = json::array();
    TypeTable& type_table_;

    DefMap<std::string> known_defs;

    std::string translate_def (const Def * def) {
        auto it = known_defs.find(def);
        if (it != known_defs.end()) {
            return it->second;
        }

        if (def->isa<MemOp>()) {
            std::stack<const Def*> required_defs;
            std::queue<const Def*> todo;
            todo.push(def);

            while (!todo.empty()) {
                auto def = todo.front();
                todo.pop();
                if (known_defs.lookup(def)) continue;

                if (auto memop = def->isa<MemOp>()) {
                    todo.push(memop->mem());
                    required_defs.push(memop->mem());
                } else if (auto extract = def->isa<Extract>()) {
                    if (is_mem(extract)) {
                        todo.push(extract->agg());
                        required_defs.push(extract->agg());
                    }
                }
            }

            while (!required_defs.empty()) {
                auto r = pop(required_defs);
                translate_def(r);
            }
        }

        json result;
        if (auto cont = def->isa<Continuation>()) {
            if (cont->is_intrinsic()) {
                if (cont->intrinsic() == Intrinsic::Branch) {
                    result["name"] = "branch";
                    result["type"] = "continuation";
                    result["intrinsic"] = "branch";
                } else if (cont->intrinsic() == Intrinsic::Match) {
                    size_t num_patterns = cont->num_params() - 3;
                    auto variant_type = type_table_.translate_type(cont->param(1)->type());
                    auto name = "_match_" + std::to_string(def_table.size());

                    result["name"] = name;
                    result["type"] = "continuation";
                    result["intrinsic"] = "match";
                    result["variant_type"] = variant_type;
                    result["num_patterns"] = num_patterns;
                } else {
                    auto intrinsic_name = cont->name();
                    auto intrinsic_type = type_table_.translate_type(cont->type());
                    auto name = "_in_" + std::to_string(def_table.size());

                    result["name"] = name;
                    result["type"] = "continuation";
                    result["intrinsic"] = intrinsic_name;
                    result["fn_type"] = intrinsic_type;
                }
                if (cont->filter() && !cont->filter()->empty())
                    result["filter"] = translate_def(cont->filter());
            } else {
                auto type = type_table_.translate_type(def->type());

                //Make the name available in known_defs as early as possible to prevent recursion issues.
                auto name = "_cont_" + std::to_string(decl_table.size());
                known_defs[def] = name;

                //TODO: Is this actually required for imported functions?
                json arg_names = json::array();
                for (auto arg : cont->params()) {
                    arg_names.push_back(translate_def(arg));
                }

                json forward_decl;
                forward_decl["name"] = name;
                forward_decl["type"] = "continuation";
                forward_decl["fn_type"] = type;
                forward_decl["arg_names"] = arg_names;
                if (cont->is_external()) {
                    if (cont->cc() == CC::Thorin)
                        forward_decl["internal"] = cont->name();
                    else
                        forward_decl["external"] = cont->name();
                }
                decl_table.push_back(forward_decl);

                if(cont->has_body()) {
                    auto app = cont->body();
                    auto target = translate_def(app->callee());
                    json args = json::array();
                    for (auto arg : app->args()) {
                        args.push_back(translate_def(arg));
                    }

                    result["name"] = name;
                    result["type"] = "continuation";
                    if (cont->filter() && !cont->filter()->empty())
                        result["filter"] = translate_def(cont->filter());
                    result["app"] = {
                        {"target", target},
                        {"args", args}
                    };
                } else {
                    //Early return. We do not have a body, so there is no point in writing something to the def table.
                    if (cont->filter() && !cont->filter()->empty())
                        assert(false && "These filters cannot be generated RN");
                    return name;
                }
            }
        } else if (auto lit = def->isa<PrimLit>()) {
            auto name = "_" + std::to_string(def_table.size());
            auto type = type_table_.translate_type(lit->type());

            result["name"] = name;
            result["type"] = "const";
            result["const_type"] = type;
            //result["value"] = lit->value().get_s32(); //TODO: this looks wrong. What I get should depend on the lit type.
            switch (lit->primtype_tag()) {
#define THORIN_I_TYPE(T, M) case PrimType_##T: { result["value"] = lit->value().get_##M(); break; }
#define THORIN_BOOL_TYPE(T, M) case PrimType_##T: { result["value"] = lit->value().get_##M(); break; }
#define THORIN_F_TYPE(T, M) case PrimType_##T: { result["value"] = (double)lit->value().get_##M(); break; }
#include <thorin/tables/primtypetable.h>
            default:
                assert(false && "not implemented");
            }
        } else if (def->isa<Top>()) {
            auto name = "_" + std::to_string(def_table.size());
            auto type = type_table_.translate_type(def->type());

            result["name"] = name;
            result["type"] = "top";
            result["const_type"] = type;
        } else if (def->isa<Bottom>()) {
            auto name = "_" + std::to_string(def_table.size());
            auto type = type_table_.translate_type(def->type());

            result["name"] = name;
            result["type"] = "bottom";
            result["const_type"] = type;
        } else if (auto param = def->isa<Param>()) {
            auto name = param->continuation()->unique_name() + "." + std::to_string(param->index());
            known_defs[def] = name;
            return name;
        } else if (auto load = def->isa<Load>()) {
            json args = json::array();
            args.push_back(translate_def(load->mem()));
            args.push_back(translate_def(load->ptr()));
            auto name = "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "load";
            result["args"] = args;
        } else if (auto store = def->isa<Store>()) {
            json args = json::array();
            args.push_back(translate_def(store->mem()));
            args.push_back(translate_def(store->ptr()));
            args.push_back(translate_def(store->val()));
            auto name = "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "store";
            result["args"] = args;
        } else if (auto size_of = def->isa<SizeOf>()) {
            auto target_type = type_table_.translate_type(size_of->of());
            auto name = "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "sizeof";
            result["target_type"] = target_type;
        } else if (auto align_of = def->isa<AlignOf>()) {
            auto target_type = type_table_.translate_type(align_of->of());
            auto name = "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "alignof";
            result["target_type"] = target_type;
        } else if (auto cast = def->isa<Cast>()) {
            auto source = translate_def(cast->from());
            auto target_type = type_table_.translate_type(cast->type());
            auto name = "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "cast";
            result["source"] = source;
            result["target_type"] = target_type;
         } else if (auto bitcast = def->isa<Bitcast>()) {
            auto source = translate_def(bitcast->from());
            auto target_type = type_table_.translate_type(bitcast->type());
            auto name = "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "bitcast";
            result["source"] = source;
            result["target_type"] = target_type;
        } else if (auto indef_array = def->isa<IndefiniteArray>()) {
            auto dim = translate_def(indef_array->op(0));
            auto name = "_" + std::to_string(def_table.size());
            auto element_type = type_table_.translate_type(indef_array->elem_type());

            result["name"] = name;
            result["type"] = "indef_array";
            result["elem_type"] = element_type;
            result["dim"] = dim;
        } else if (auto def_array = def->isa<DefiniteArray>()) {
            json args = json::array();
            for (auto arg : def_array->ops()) {
                args.push_back(translate_def(arg));
            }

            auto name = "_" + std::to_string(def_table.size());
            auto element_type = type_table_.translate_type(def_array->elem_type());

            result["name"] = name;
            result["type"] = "def_array";
            result["elem_type"] = element_type;
            result["args"] = args;
        } else if (auto lea = def->isa<LEA>()) {
            json args = json::array();
            args.push_back(translate_def(lea->ptr()));
            args.push_back(translate_def(lea->index()));
            auto name = "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "lea";
            result["args"] = args;
        } else if (auto extract = def->isa<Extract>()) {
            json args = json::array();
            args.push_back(translate_def(extract->agg()));
            args.push_back(translate_def(extract->index()));
            auto name = "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "extract";
            result["args"] = args;
        } else if (auto insert = def->isa<Insert>()) {
            json args = json::array();
            args.push_back(translate_def(insert->agg()));
            args.push_back(translate_def(insert->index()));
            args.push_back(translate_def(insert->value()));
            auto name = "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "insert";
            result["args"] = args;
        } else if (auto closure = def->isa<Closure>()) {
            json args = json::array();
            args.push_back(translate_def(closure->op(0)));
            args.push_back(translate_def(closure->op(1)));
            auto closure_type = type_table_.translate_type(closure->type());
            auto name = "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "closure";
            result["args"] = args;
            result["closure_type"] = closure_type;
        } else if (auto struct_agg = def->isa<StructAgg>()) {
            json args = json::array();
            for (auto arg : struct_agg->ops()) {
                args.push_back(translate_def(arg));
            }
            auto struct_type = type_table_.translate_type(struct_agg->type());
            auto name = "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "struct";
            result["args"] = args;
            result["struct_type"] = struct_type;
        } else if (auto tuple = def->isa<Tuple>()) {
            json args = json::array();
            for (auto arg : tuple->ops()) {
                args.push_back(translate_def(arg));
            }
            auto name = "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "tuple";
            result["args"] = args;
        } else if (auto vector = def->isa<Vector>()) {
            json args = json::array();
            for (auto arg : vector->ops()) {
                args.push_back(translate_def(arg));
            }
            auto name = "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "vector";
            result["args"] = args;
        } else if (auto filter = def->isa<Filter>()) {
            json args = json::array();
            for (auto arg : filter->ops()) {
                args.push_back(translate_def(arg));
            }
            auto name = "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "filter";
            result["args"] = args;
        } else if (auto arithop = def->isa<ArithOp>()) {
            auto op = arithop->op_name();
            json args = json::array();
            args.push_back(translate_def(arithop->lhs()));
            args.push_back(translate_def(arithop->rhs()));
            auto name = "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "arithop";
            result["op"] = op;
            result["args"] = args;
        } else if (auto mathop = def->isa<MathOp>()) {
            auto op = mathop->op_name();
            json args = json::array();
            for (auto arg : mathop->ops()) {
                args.push_back(translate_def(arg));
            }
            auto name = "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "mathop";
            result["op"] = op;
            result["args"] = args;
        } else if (auto select = def->isa<Select>()) {
            json args = json::array();
            args.push_back(translate_def(select->cond()));
            args.push_back(translate_def(select->tval()));
            args.push_back(translate_def(select->fval()));
            auto name = "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "select";
            result["args"] = args;
        } else if (auto cmp = def->isa<Cmp>()) {
            auto op = cmp->op_name();
            json args = json::array();
            args.push_back(translate_def(cmp->lhs()));
            args.push_back(translate_def(cmp->rhs()));
            auto name = "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "cmp";
            result["op"] = op;
            result["args"] = args;
        } else if (auto run = def->isa<Run>()) {
            auto target = translate_def(run->def());
            auto name = "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "run";
            result["target"] = target;
        } else if (auto hlt = def->isa<Hlt>()) {
            auto target = translate_def(hlt->def());
            auto name = "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "hlt";
            result["target"] = target;
        } else if (auto known = def->isa<Known>()) {
            auto def = translate_def(known->def());
            auto name = "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "known";
            result["def"] = def;
        } else if (auto enter = def->isa<Enter>()) {
            auto mem = translate_def(enter->mem());
            auto name = "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "enter";
            result["mem"] = mem;
        } else if (auto slot = def->isa<Slot>()) {
            auto frame = translate_def(slot->frame());
            auto target_type = type_table_.translate_type(slot->alloced_type());
            auto name = "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "slot";
            result["frame"] = frame;
            result["target_type"] = target_type;
        } else if (auto alloc = def->isa<Alloc>()) {
            json args = json::array();
            args.push_back(translate_def(alloc->mem()));
            args.push_back(translate_def(alloc->extra()));
            auto target_type = type_table_.translate_type(alloc->alloced_type());
            auto name = "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "alloc";
            result["args"] = args;
            result["target_type"] = target_type;
        } else if (auto global = def->isa<Global>()) {
            auto init = translate_def(global->init());
            bool is_mutable = global->is_mutable();
            bool is_external = global->is_external();
            auto name = "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "global";
            result["init"] = init;
            result["mutable"] = is_mutable;
            if (is_external)
                result["external"] = global->name();
        } else if (auto variant = def->isa<Variant>()) {
            auto variant_type = type_table_.translate_type(variant->type());
            auto value = translate_def(variant->value());
            size_t index = variant->index();
            auto name = "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "variant";
            result["variant_type"] = variant_type;
            result["value"] = value;
            result["index"] = index;
        } else if (auto variant_extract = def->isa<VariantExtract>()) {
            auto value = translate_def(variant_extract->value());
            size_t index = variant_extract->index();
            auto name = "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "variant_extract";
            result["value"] = value;
            result["index"] = index;
        } else if (auto variant_index = def->isa<VariantIndex>()) {
            auto value = translate_def(variant_index->op(0));
            auto name = "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "variant_index";
            result["value"] = value;
        } else if (auto assembly = def->isa<Assembly>()) {
            auto asm_type = type_table_.translate_type(assembly->type());
            json inputs = json::array();
            inputs.push_back(translate_def(assembly->mem()));
            for (auto input : assembly->inputs()) {
                inputs.push_back(translate_def(input));
            }
            auto asm_template = assembly->asm_template();
            json out_constraints = json::array();
            for (auto constraint : assembly->output_constraints()) {
                out_constraints.push_back(constraint);
            }
            json in_constraints = json::array();
            for (auto constraint : assembly->input_constraints()) {
                in_constraints.push_back(constraint);
            }
            json clobbers = json::array();
            for (auto c : assembly->clobbers()) {
                clobbers.push_back(c);
            }

            auto name = "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "assembly";

            result["asm_type"] = asm_type;
            result["inputs"] = inputs;
            result["asm_template"] = asm_template;

            result["output_constraints"] = out_constraints;
            result["input_constraints"] = in_constraints;
            result["clobbers"] = clobbers;
            switch (assembly->flags()) {
            case Assembly::Flags::NoFlag:
                result["flags"] = "noflag";
                break;
            case Assembly::Flags::HasSideEffects:
                result["flags"] = "hassideeffects";
                break;
            case Assembly::Flags::IsAlignStack:
                result["flags"] = "isalignstack";
                break;
            case Assembly::Flags::IsIntelDialect:
                result["flags"] = "isinteldialect";
                break;
            }
        } else {
            def->dump();
            def->dump(2);
            std::cerr << "cannot be translated\n";
            THORIN_UNREACHABLE;
        }
        known_defs[def] = result["name"];
        def_table.push_back(result);
        return result["name"];
    }
};

void CodeGen::emit_json(json& j) {
    j["module"] = world().name();
    if (target_triple != "")
        j["target_triple"] = target_triple;
    if (target_cpu != "")
        j["target_cpu"] = target_cpu;
    if (target_attr != "")
        j["target_attr"] = target_attr;

    TypeTable type_table;
    DefTable def_table(type_table);

    for (auto external : world().externals()) {
        def_table.translate_def(external.second);
    }

    j["type_table"] = type_table.nominal_fwd_table;
    for (auto it : type_table.type_table)
        j["type_table"] += it;

    j["defs"] = def_table.decl_table;
    for (auto it : def_table.def_table)
        j["defs"] += it;
}

void CodeGen::emit_stream(std::ostream& stream) {
    json j;

    emit_json(j);
    
    Stream s(stream);
    s << j.dump(2) << "\n";
}

}
