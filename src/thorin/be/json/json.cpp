#include "json.h"

namespace thorin::json {

class TypeTable {
public:
    json type_table = json::array();

    TypeMap<std::string> known_types;

    std::string translate_type (const Type * type) {
        auto it = known_types.find(type);
        if (it != known_types.end()) {
            return it->second;
        }

        json result;
        if (type->isa<MemType>()) {
            result["name"] = "mem_t";
            result["type"] = "mem";
        } else if (auto prim = type->isa<PrimType>()) {
            result["name"] = "_" + std::to_string(type_table.size());
            result["length"] = prim->length();
            result["type"] = "prim";
            switch (prim->primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case PrimTypeTag::PrimType_##T: { result["tag"] = #T; break; }
#include <thorin/tables/primtypetable.h>
            }
        } else if (auto fntype = type->isa<FnType>()) {
            json arg_types = json::array();
            for (auto arg : fntype->ops()) {
                arg_types.push_back(translate_type(arg));
            }

            result["type"] = "fn";
            result["name"] = "_" + std::to_string(type_table.size());
            result["args"] = arg_types;
        } else if (auto ptrtype = type->isa<PtrType>()) {
            auto pointee_type = translate_type(ptrtype->pointee());

            result["type"] = "ptr";
            result["args"] = { pointee_type };
            result["name"] = pointee_type + "_p";
            result["length"] = ptrtype->length();
        } else if (auto arr = type->isa<IndefiniteArrayType>()) {
            auto elem_type = translate_type(arr->elem_type());

            result["type"] = "indef_array";
            result["args"] = { elem_type };
            result["name"] = elem_type + "_iarr";
        } else {
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

    std::string translate_def (const Def * def, std::string expected_name = "") {
        json result;
        if (auto cont = def->isa<Continuation>()) {
            auto type = type_table_.translate_type(def->type());
            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());
            json arg_names;
            for (auto arg : cont->params()) {
                arg_names.push_back(translate_def(arg));
            }

            json forward_decl;
            forward_decl["name"] = name;
            forward_decl["type"] = "continuation";
            forward_decl["fn_type"] = type;
            forward_decl["arg_names"] = arg_names;
            forward_decl["external"] = cont->is_external();
            decl_table.push_back(forward_decl); //TODO: should be pushed to the front instead.

            assert(cont->has_body());
            auto app = cont->body();
            auto target = translate_def(app->callee());
            json args = json::array();
            for (auto arg : app->args()) {
                args.push_back(translate_def(arg));
            }

            result["name"] = name;
            result["type"] = "continuation";
            result["app"] = {
                {"target", target},
                {"args", args}
            };
        } else if (auto lit = def->isa<PrimLit>()) {
            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());
            auto type = type_table_.translate_type(def->type());

            result["name"] = name;
            result["type"] = "const";
            result["const_type"] = type;
            result["value"] = lit->value().get_s32(); //TODO: this looks wrong. What I get should depend on the lit type.
        } else if (auto param = def->isa<Param>()) {
            auto name = expected_name != "" ? expected_name : param->continuation()->unique_name() + "." + std::to_string(param->index());
            known_defs[def] = name;
            return name;
        } else {
            THORIN_UNREACHABLE;
        }
        known_defs[def] = result["name"];
        def_table.push_back(result);
        return result["name"];
    }
};

void CodeGen::emit_stream(std::ostream& stream) {
    json j;

    j["module"] = world().name();

    TypeTable type_table;
    DefTable def_table(type_table);

    for (auto external : world().externals()) {
        const Continuation* continuation = external.second;
        auto expected_name = continuation->name();
        def_table.translate_def(continuation, expected_name);
    }

    j["type_table"] = type_table.type_table;
    j["defs"] = def_table.decl_table;
    for (auto it : def_table.def_table)
        j["defs"] += it;
    
    std::cerr << j.dump(2) << std::endl;

    Stream s(stream);
    s << j.dump(2) << "\n";
}

}
