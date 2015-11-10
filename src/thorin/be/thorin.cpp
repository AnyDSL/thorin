#include "thorin/be/thorin.h"

#include "thorin/lambda.h"
#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/printer.h"

namespace thorin {

class CodeGen : public Printer {
public:
    CodeGen(std::ostream& ostream)
        : Printer(ostream)
    {}

    std::ostream& emit_type_vars(Type);
    std::ostream& emit_type_args(Type);
    std::ostream& emit_type_elems(Type);
    std::ostream& emit_type(Type);
    std::ostream& emit_name(Def);
    std::ostream& emit_def(Def);
    std::ostream& emit_primop(const PrimOp*);
    std::ostream& emit_assignment(const PrimOp*);
    std::ostream& emit_head(const Lambda*);
    std::ostream& emit_jump(const Lambda*);
};

//------------------------------------------------------------------------------

std::ostream& CodeGen::emit_type_vars(Type type) {
    if (type->num_type_vars() != 0)
        return dump_list([&](TypeVar type_var) { emit_type(type_var); }, type->type_vars(), "[", "]");
    return stream();
}

std::ostream& CodeGen::emit_type_args(Type type) {
    return dump_list([&](Type type) { emit_type(type); }, type->args(), "(", ")");
}

std::ostream& CodeGen::emit_type_elems(Type type) {
    if (auto struct_app = type.isa<StructAppType>())
        return dump_list([&](Type type) { emit_type(type); }, struct_app->elems(), "{", "}");
    return emit_type_args(type);
}

std::ostream& CodeGen::emit_type(Type type) {
    if (type.empty()) {
        return stream() << "<NULL>";
    } else if (type.isa<FrameType>()) {
        return stream() << "frame";
    } else if (type.isa<MemType>()) {
        return stream() << "mem";
    } else if (auto fn = type.isa<FnType>()) {
        stream() << "fn";
        emit_type_vars(fn);
        return emit_type_args(fn);
    } else if (auto tuple = type.isa<TupleType>()) {
        emit_type_vars(tuple);
        return emit_type_args(tuple);
    } else if (auto struct_abs = type.isa<StructAbsType>()) {
        stream() << struct_abs->name();
        return emit_type_vars(struct_abs);
        // TODO emit args - but don't do this inline: structs may be recursive
        //return emit_type_args(struct_abs);
    } else if (auto struct_app = type.isa<StructAppType>()) {
        stream() << struct_app->struct_abs_type()->name();
        return emit_type_elems(struct_app);
    } else if (auto type_var = type.isa<TypeVar>()) {
        return stream() << '<' << type_var->gid() << '>';
    } else if (auto array = type.isa<IndefiniteArrayType>()) {
        stream() << '[';
        emit_type(array->elem_type());
        return stream() << ']';
    } else if (auto array = type.isa<DefiniteArrayType>()) {
        stream() << '[' << array->dim() << " x ";
        emit_type(array->elem_type());
        return stream() << ']';
    } else if (auto ptr = type.isa<PtrType>()) {
        if (ptr->is_vector())
            stream() << '<' << ptr->length() << " x ";
        emit_type(ptr->referenced_type());
        stream() << '*';
        if (ptr->is_vector())
            stream() << '>';
        auto device = ptr->device();
        if (device != -1)
            stream() << '[' << device << ']';
        switch (ptr->addr_space()) {
            case AddressSpace::Global:   stream() << "[Global]";   break;
            case AddressSpace::Texture:  stream() << "[Tex]";      break;
            case AddressSpace::Shared:   stream() << "[Shared]";   break;
            case AddressSpace::Constant: stream() << "[Constant]"; break;
            default: /* ignore unknown address space */            break;
        }
        return stream();
    } else if (auto primtype = type.isa<PrimType>()) {
        if (primtype->is_vector())
            stream() << "<" << primtype->length() << " x ";
            switch (primtype->primtype_kind()) {
#define THORIN_ALL_TYPE(T, M) case Node_PrimType_##T: stream() << #T; break;
#include "thorin/tables/primtypetable.h"
                default: THORIN_UNREACHABLE;
            }
        if (primtype->is_vector())
            stream() << ">";
        return stream();
    }
    THORIN_UNREACHABLE;
}

std::ostream& CodeGen::emit_def(Def def) {
    if (auto primop = def->isa<PrimOp>())
        return emit_primop(primop);
    return emit_name(def);
}

std::ostream& CodeGen::emit_name(Def def) {
    return stream() << (def->isa<Lambda>() && def->as<Lambda>()->is_intrinsic() ? def->name : def->unique_name());
}

std::ostream& CodeGen::emit_primop(const PrimOp* primop) {
    if (primop->is_proxy())
        stream() << "<proxy>";
    else if (auto primlit = primop->isa<PrimLit>()) {
        emit_type(primop->type()) << ' ';
        auto kind = primlit->primtype_kind();

        // print i8 as ints
        if (kind == PrimType_qs8)
            stream() << (int) primlit->qs8_value();
        else if (kind == PrimType_ps8)
            stream() << (int) primlit->ps8_value();
        else if (kind == PrimType_qu8)
            stream() << (unsigned) primlit->qu8_value();
        else if (kind == PrimType_pu8)
            stream() << (unsigned) primlit->pu8_value();
        else {
            switch (kind) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: stream() << primlit->T##_value(); break;
#include "thorin/tables/primtypetable.h"
                default: THORIN_UNREACHABLE;
            }
        }
    } else if (primop->isa<Global>()) {
        emit_name(primop);
    } else if (primop->is_const()) {
        if (primop->empty()) {
            stream() << primop->op_name() << ' ';
            emit_type(primop->type());
        } else {
            stream() << '(';
            if (primop->isa<PrimLit>())
                emit_type(primop->type()) << ' ';
            stream() << primop->op_name();
            dump_list([&](Def def) { emit_def(def); }, primop->ops(), " ", ")");
        }
    } else
        emit_name(primop);

    return stream();
}

std::ostream& CodeGen::emit_assignment(const PrimOp* primop) {
    emit_type(primop->type()) << " ";
    emit_name(primop) << " = ";
    stream() << primop->op_name() << " ";
    dump_list([&](Def def) { emit_def(def); }, primop->ops());
    return newline();
}

std::ostream& CodeGen::emit_head(const Lambda* lambda) {
    emit_name(lambda);
    emit_type_vars(lambda->type());
    dump_list([&](const Param* param) { emit_type(param->type()) << " "; emit_name(param); }, lambda->params(), "(", ")");

    if (lambda->is_external())
        stream() << " extern ";
    if (lambda->cc() == CC::Device)
        stream() << " device ";

    return up();
}

std::ostream& CodeGen::emit_jump(const Lambda* lambda) {
    if (!lambda->empty()) {
        emit_def(lambda->to());
        dump_list([&](Def def) { emit_def(def); }, lambda->args(), " ", "");
    }
    return down();
}

//------------------------------------------------------------------------------

std::ostream& emit_thorin(const Scope& scope, std::ostream& ostream) {
    CodeGen cg(ostream);
    auto schedule = schedule_smart(scope);
    for (auto& block : schedule) {
        auto lambda = block.lambda();
        if (lambda->intrinsic() != Intrinsic::EndScope) {
            int depth = lambda == scope.entry() ? 0 : 1;
            cg.indent += depth;
            cg.newline();
            cg.emit_head(lambda);

            for (auto primop : block)
                cg.emit_assignment(primop);

            cg.emit_jump(lambda);
            cg.indent -= depth;
        }
    }
    return cg.newline();
}

std::ostream& emit_thorin(const World& world, std::ostream& ostream) {
    CodeGen cg(ostream);
    cg.stream() << "module '" << world.name() << "'\n\n";

    for (auto primop : world.primops()) {
        if (auto global = primop->isa<Global>())
            cg.emit_assignment(global);
    }

    Scope::for_each<false>(world, [&] (const Scope& scope) { emit_thorin(scope, ostream); });
    return ostream;
}

std::ostream& emit_type(Type type, std::ostream& ostream)                  { return CodeGen(ostream).emit_type(type);         }
std::ostream& emit_def(Def def, std::ostream& ostream)                     { return CodeGen(ostream).emit_def(def);           }
std::ostream& emit_name(Def def, std::ostream& ostream)                    { return CodeGen(ostream).emit_name(def);          }
std::ostream& emit_head(const Lambda* lambda, std::ostream& ostream)       { return CodeGen(ostream).emit_head(lambda);       }
std::ostream& emit_jump(const Lambda* lambda, std::ostream& ostream)       { return CodeGen(ostream).emit_jump(lambda);       }
std::ostream& emit_assignment(const PrimOp* primop, std::ostream& ostream) { return CodeGen(ostream).emit_assignment(primop); }

//------------------------------------------------------------------------------

void Scope::stream_thorin(std::ostream& out) const { emit_thorin(*this, out); }
void World::stream_thorin(std::ostream& out) const { emit_thorin(*this, out); }

void Scope::dump() const { emit_thorin(*this, std::cout); }
void World::dump() const { emit_thorin(*this, std::cout); }

void Scope::write_thorin(const char* filename) const { std::ofstream file(filename); stream_thorin(file); }
void World::write_thorin(const char* filename) const { std::ofstream file(filename); stream_thorin(file); }

void Scope::thorin() const {
    auto filename = world().name() + "_" + entry()->unique_name() + ".thorin";
    write_thorin(filename.c_str());
}

void World::thorin() const {
    auto filename = name() + ".thorin";
    write_thorin(filename.c_str());
}

//------------------------------------------------------------------------------

}
