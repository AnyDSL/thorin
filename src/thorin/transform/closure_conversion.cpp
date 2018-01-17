#include "thorin/continuation.h"
#include "thorin/world.h"
#include "thorin/analyses/verify.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/log.h"

namespace thorin {

/*
class ClosureConversion {
public:
    ClosureConversion(World& world)
        : world_(world)
    {}

    void run() {
        Scope::for_each(world, [&] (Scope& scope) {
            bool dirty = false;
            for (auto n : scope.f_cfg().post_order()) {
                auto callee = n.continuation()->callee();
                if (callee->isa<Closure> || callee->isa_continuation())
                    continue;
                
            }
            if (dirty)
                scope.update();
        });
        for (auto cont : world_.copy_continuations()) {
            if (cont->is_basicblock() || cont->is_returning() || cont->is_intrinsic())
                continue;

            const Type* new_type = convert(cont->type());
            if (auto ptr_type = new_type->isa<PtrType>())
                new_type = ptr_type->pointee();
            auto new_cont = world_.continuation(new_type->as<FnType>(), cont->debug());
            new_cont->dump_head();
            for (size_t i = 0, e = new_cont->num_params(); i != e; ++i)
                new_defs_.emplace(cont->param(i), new_cont->param(i));
            new_defs_.emplace(cont, new_cont);
        }

        // inspect each scope and remove non function calls
        Scope::for_each(world_, [&] (Scope& scope) {
            for (auto cont : scope) {
                Array<const Def*> args(cont->args());
                for (auto& arg : args) arg = convert(arg);
                cont->jump(convert(cont->callee()), args);
                scope.update();
            }
        });
        world_.dump();
        exit(0);
    }

    const Def* convert(const Def* def) {
        if (new_defs_.count(def)) return new_defs_[def];
        if (auto primop = def->isa<PrimOp>()) {
            Array<const Def*> ops(primop->ops());
            for (auto& op : ops) op = convert(op);
            return new_defs_[def] = primop->rebuild(ops, convert(primop->type()));
        } else {
            return new_defs_[def] = def;
        }
    }

    // convert functions to function pointers
    // - fn (A, B, fn(C), fn(D)) => closure(fn (A, B, fn(convert(C)), closure(fn(convert(D)))))
    // - struct S { fn (X, fn(Y)) } => struct T { closure(fn (X, fn(Y))) }
    // - ...
    const Type* convert(const Type* type) {
        if (new_types_.count(type)) return new_types_[type];
        if (type->order() <= 1) return type;
        Array<const Type*> ops(type->ops());
        // accept one parameter of order 1 (the return continuation) for function types
        bool ret = !type->isa<FnType>();
        for (auto& op : ops) {
            op = convert(op);
            if (!ret &&
                op->isa<ClosureType>() &&
                op->as<ClosureType>()->pointee()->order() == 1) {
                ret = true;
                op = op->as<ClosureType>()->pointee();
            }
        }
        const Type* new_type = nullptr;
        if (type->isa<StructType>()) {
            // struct types cannot be rebuilt 
            auto struct_type =  world_.struct_type(type->as<StructType>()->name(), ops.size());
            for (size_t i = 0, e = ops.size(); i != e; ++i)
                struct_type->set(i, ops[i]);
            new_type = struct_type;
        } else {
            new_type = type->rebuild(ops);
        }
        if (new_type->order() <= 1)
            return new_types_[type] = new_type;
        else
            return new_types_[type] = world_.closure_type(new_type);
    }

private:
    World& world_;
    Def2Def new_defs_;
    Type2Type new_types_;
};
*/

void closure_conversion(World& world) {
    //ClosureConversion(world).run();
}

}
