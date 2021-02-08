#include "thorin/be/f95.h"
#include "thorin/world.h"

#include <cmath>
#include <sstream>
#include <type_traits>
#include <ostream>
#include <cctype>

namespace thorin {

class F95CodeGen {
public:
    F95CodeGen(World& world, std::ostream& stream)
      : world_(world)
      , os_(stream)
    {}

    void emit_f95_int();
    World& world() const { return world_; }
private:
    std::ostream& emit_aggop_decl(const Type* type);
    std::ostream& emit_type(std::ostream& os, const Type* type);

    bool lookup(const Type*);
    bool lookup(const Def*);
    void insert(const Type*, std::string);
    void insert(const Def*, std::string);
    std::string& get_name(const Type* type);
    std::string& get_name(const Def* def);
    const std::string var_name(const Def* def);
    std::string type_name(const Type* type);
    std::string array_name(const DefiniteArrayType* array_type);
    std::string tuple_name(const TupleType* tuple_type);

    World& world_;
    TypeMap<std::string> type2str_;
    DefMap<std::string> def2str_;
    DefMap<std::string> global2str_;
    DefMap<std::string> primop2str_;
    int primop_counter = 0;
    std::ostream& os_;
    std::ostringstream func_impl_;
    std::ostringstream func_decls_;
    std::ostringstream type_decls_;
};

//------------------------------------------------------------------------------

inline std::string make_identifier(const std::string& str) {
    auto copy = str;
    for (auto& c : copy) {
        if (c == ' ') c = '_';
    }
    return copy;
}

//------------------------------------------------------------------------------

bool F95CodeGen::lookup(const Type* type) {
    return type2str_.contains(type);
}

bool F95CodeGen::lookup(const Def* def) {
    if (def->isa<Global>())
        return global2str_.contains(def);
    else if (def->isa<PrimOp>() && is_const(def))
        return primop2str_.contains(def);
    else
        return def2str_.contains(def);
}

std::string& F95CodeGen::get_name(const Type* type) {
    return type2str_[type];
}

std::string& F95CodeGen::get_name(const Def* def) {
    if (def->isa<Global>())
        return global2str_[def];
    else if (def->isa<PrimOp>() && is_const(def))
        return primop2str_[def];
    else
        return def2str_[def];
}

const std::string F95CodeGen::var_name(const Def* def) {
    if (def->isa<PrimOp>() && is_const(def))
        return def->unique_name() + "_" + std::to_string(primop_counter++);
    else
        return def->unique_name();
}

void F95CodeGen::insert(const Type* type, std::string str) {
    type2str_[type] = str;
}

void F95CodeGen::insert(const Def* def, std::string str) {
    if (def->isa<Global>())
        global2str_[def] = str;
    else if (def->isa<PrimOp>() && is_const(def))
        primop2str_[def] = str;
    else
        def2str_[def] = str;
}

std::ostream& F95CodeGen::emit_type(std::ostream& os, const Type* type) {
    if (lookup(type))
        return os << get_name(type);

    if (type == nullptr) {
        return os << "NULL()";
    } else if (type->isa<FrameType>()) {
        return os;
    } else if (type->isa<MemType>() || type == world().unit()) {
        return os << "!! TODO:VOID !!";
    } else if (type->isa<FnType>()) {
        THORIN_UNREACHABLE;
    } else if (auto tuple = type->isa<TupleType>()) {
        os << "TYPE " << tuple_name(tuple) << endl;
        os << "SEQUENCE" << up;
        for (size_t i = 0, e = tuple->ops().size(); i != e; ++i) {
            os << endl;
            emit_type(os, tuple->op(i)) << " :: e" << i;
        }
        return os << down << endl << "END TYPE " << tuple_name(tuple);
    } else if (auto variant = type->isa<VariantType>()) {
        THORIN_UNREACHABLE;
    } else if (auto struct_type = type->isa<StructType>()) {
        os << "TYPE " << struct_type->name() << endl;
        os << "SEQUENCE" << up;
        for (size_t i = 0, e = struct_type->num_ops(); i != e; ++i) {
            os << endl;
            emit_type(os, struct_type->op(i)) << " :: " << struct_type->op_name(i);
        }
        return os << down << endl << "END TYPE " << struct_type->name();
    } else if (auto array = type->isa<IndefiniteArrayType>()) {
        emit_type(os, array->elem_type());
        return os;
    } else if (auto array = type->isa<DefiniteArrayType>()) { // DefArray is mapped to a struct
        emit_type(os, array->elem_type()) << " ";

        std::stringstream elem_name;
        emit_type(elem_name, array->elem_type());
        return os << make_identifier(elem_name.str()) << "(" << array->dim() << ")";
    } else if (auto ptr = type->isa<PtrType>()) {
        emit_type(os, ptr->pointee());
        os << ", POINTER ";
        if (ptr->is_vector())
            THORIN_UNREACHABLE;
        return os;
    } else if (auto primtype = type->isa<PrimType>()) {
        switch (primtype->primtype_tag()) {
            case PrimType_bool:                     os << "LOGICAL(1)";  break;
            case PrimType_ps8:  case PrimType_qs8:  os << "INTEGER(1)";  break;
            case PrimType_pu8:  case PrimType_qu8:  os << "UNSIGNED(1)"; break;
            case PrimType_ps16: case PrimType_qs16: os << "INTEGER(2)";  break;
            case PrimType_pu16: case PrimType_qu16: os << "UNSIGNED(2)"; break;
            case PrimType_ps32: case PrimType_qs32: os << "INTEGER(4)";  break;
            case PrimType_pu32: case PrimType_qu32: os << "UNSIGNED(4)"; break;
            case PrimType_ps64: case PrimType_qs64: os << "INTEGER(8)";  break;
            case PrimType_pu64: case PrimType_qu64: os << "UNSIGNED(8)"; break;
            case PrimType_pf16: case PrimType_qf16: os << "REAL(4)";     break;
            case PrimType_pf32: case PrimType_qf32: os << "REAL(4)";     break;
            case PrimType_pf64: case PrimType_qf64: os << "REAL(8)";     break;
        }
        if (primtype->is_vector())
            THORIN_UNREACHABLE;
        return os;
    }
    THORIN_UNREACHABLE;
}

std::ostream& F95CodeGen::emit_aggop_decl(const Type* type) {
    if (lookup(type) || type == world().unit())
        return type_decls_;

    // set indent to zero
    auto indent = detail::get_indent();
    while (detail::get_indent() != 0)
        type_decls_ << down;

    if (auto ptr = type->isa<PtrType>())
        emit_aggop_decl(ptr->pointee());

    if (auto array = type->isa<IndefiniteArrayType>())
        emit_aggop_decl(array->elem_type());

    if (auto fn = type->isa<FnType>())
        for (auto type : fn->ops())
            emit_aggop_decl(type);

    // look for nested array
    if (auto array = type->isa<DefiniteArrayType>()) {
        emit_aggop_decl(array->elem_type());
        emit_type(type_decls_, array) << endl;
        insert(type, array_name(array));
    }

    // look for nested tuple
    if (auto tuple = type->isa<TupleType>()) {
        for (auto op : tuple->ops())
            emit_aggop_decl(op);
        emit_type(type_decls_, tuple) << endl;
        insert(type, tuple_name(tuple));
    }

    // look for nested struct
    if (auto struct_type = type->isa<StructType>()) {
        for (auto op : struct_type->ops())
            emit_aggop_decl(op);
        emit_type(type_decls_, struct_type) << endl;
        insert(type, struct_type->name().str());
    }

    // look for nested variants
    if (auto variant = type->isa<VariantType>()) {
        for (auto op : variant->ops())
            emit_aggop_decl(op);
        emit_type(type_decls_, variant) << endl;
        insert(type, variant->name().str());
    }

    // restore indent
    while (detail::get_indent() != indent)
        type_decls_ << up;

    return type_decls_;
}

void F95CodeGen::emit_f95_int() {
    world().cleanup();
    
    for (auto continuation : world().continuations()) {
        if (!continuation->is_imported() && !continuation->is_exported())
            continue;
        assert(continuation->is_returning());

        // retrieve return param
        const Param* ret_param = nullptr;
        for (auto param : continuation->params()) {
            if (param->order() != 0) {
                assert(!ret_param);
                ret_param = param;
            }
        }
        assert(ret_param);

        auto ret_param_fn_type = ret_param->type()->as<FnType>();
        auto ret_type = ret_param_fn_type->num_ops() > 2 ? world_.tuple_type(ret_param_fn_type->ops().skip_front()) : ret_param_fn_type->ops().back();
        if (continuation->is_imported()) {
            emit_aggop_decl(ret_type);
            for (auto param : continuation->params()) {
                if (is_mem(param) || is_unit(param) || param->order() != 0)
                    continue;
                emit_aggop_decl(param->type());
            }
            continue;
        }

        // emit function declaration
        emit_aggop_decl(ret_type);
        emit_type(func_decls_, ret_type) << " " << continuation->name() << "(";
        size_t i = 0;

        // emit and store all first-order params
        for (auto param : continuation->params()) {
            if (is_mem(param) || is_unit(param))
                continue;
            if (param->order() == 0) {
                emit_aggop_decl(param->type());
                if (i++ > 0)
                    func_decls_ << ", ";

                emit_type(func_decls_, param->type());
                insert(param, param->unique_name());
            }
        }
        func_decls_ << ");" << endl;
    }

    size_t pos = world().name().find_last_of("\\/");
    pos = (pos == std::string::npos) ? 0 : pos + 1;
    auto guard = world().name().substr(pos) + ".h";
    auto name = world().name() + ".h";

    // Generate a valid include guard macro name
    if (!std::isalpha(guard[0]) && guard[0] != '_') guard.insert(guard.begin(), '_');
    transform(guard.begin(), guard.end(), guard.begin(), [] (char c) -> char {
        if (!std::isalnum(c)) return '_';
        return ::toupper(c);
    });
    guard[guard.length() - 2] = '_';

    os_ << "! " << name << ": Artic interface file generated by thorin !" << endl;
    os_ << "#ifndef " << guard << endl;
    os_ << "#define " << guard << endl << endl;

    if (!type_decls_.str().empty())
        os_ << type_decls_.str() << endl;
    if (!func_decls_.str().empty())
        os_ << func_decls_.str() << endl;

    os_ << "#endif ! " << guard << " !";
}

std::string F95CodeGen::type_name(const Type* type) {
    std::stringstream os;
    emit_type(os, type);
    return make_identifier(std::string(os.str()));
}

std::string F95CodeGen::array_name(const DefiniteArrayType* array_type) {
    return "array_" + std::to_string(array_type->dim()) + "_" + type_name(array_type->elem_type());
}

std::string F95CodeGen::tuple_name(const TupleType* tuple_type) {
    std::string name = "tuple";
    for (auto op : tuple_type->ops())
        name += "_" + type_name(op);
    return name;
}

//------------------------------------------------------------------------------

void emit_f95_int(World& world, std::ostream& stream) {
    F95CodeGen(world, stream).emit_f95_int();
}

//------------------------------------------------------------------------------

}

