#include "anydsl2/world.h"
#include "anydsl2/type.h"
#include "anydsl2/printer.h"
#include "verifier.h"

namespace anydsl2 {

#define INVALID(def, type) { \
        invalid_.push_back(InvalidEntry(type, def)); \
        return false; \
    }
#define INVALID_CE(def, source) { \
        invalid_.push_back(InvalidEntry(def, source)); \
        return false; \
    }
#define VALID return true;

class Verifier {
public:
    Verifier(World& world, InvalidEntries& invalid)
        : world_(world)
        , invalid_(invalid)
        , pass_(world_.new_pass())
    {
        invalid_.clear();
    }

    bool verify() {
        // loop over all lambdas and check them
        for_all(lambda, world_.lambdas()) {
            verify(lambda);
        }
        return invalid_.empty();
    }

private:
    bool verify(const Def* def) {
        if(def->isa<Param>() || def->isa_lambda())
            VALID
        else
            return verify(def->as<PrimOp>());
    }

    bool verify(Lambda* lambda) {
        // check whether the lambda is stored in the world
        const LambdaSet& lambdas = world_.lambdas();
        if(lambdas.find(lambda) == lambdas.end())
            return false;
        // check the "body" of this lambda
        for_all(op, lambda->ops()) {
            // -> check the current element structure
            if(!verify(op)) INVALID_CE(lambda, op)
        }
        // there are no cycles in this body
        // => thus, we can verfiy types
        const Pi* stype = lambda->type()->isa<Pi>();
        const size_t num_params = lambda->num_params();
        // sanity check: lambda type
        if(!stype || num_params != stype->size())
            INVALID(lambda, INVALID_STRUCTURE)
        // sanity check: check each parameter type
        for(size_t i = 0; i < num_params; ++i)
            if(!stype->elem(i)->check_with(lambda->param(i)->type()))
                INVALID(lambda, INVALID_TYPES)

        // verifiy the actual call
        const Pi* ttype = lambda->to()->type()->isa<Pi>();
        const size_t num_args = lambda->num_args();
        // sanity check: number arguments for call site
        if(!ttype || num_args != ttype->size())
            INVALID(lambda, INVALID_STRUCTURE)

        // sanity check: argument types
        for(size_t i = 0; i < num_args; ++i) {
            if(!ttype->elem(i)->check_with(lambda->arg(i)->type()))
                INVALID(lambda, INVALID_TYPES)
        }

        // argument types are compatible in general
        // => check inference mapping
        GenericMap generics;
        if(!ttype->infer_with(generics, lambda->arg_pi()))
            INVALID(lambda, INVALID_TYPES_GENERICS)
        // all checks passed => seems to be a valid lambda
        return true;
    }

    bool verify(const PrimOp* primop) {
        // check whether the current primop is stored in the world

        // if we have detected a cycle -> invalid
        if(primop->is_visited(pass_) && !primop->is_const())
            INVALID(primop, INVALID_CYCLIC_DEPENDENCY)
        primop->visit(pass_);
        // check all operands recursively
        for_all(op, primop->ops()) {
            if(!verify(op)) INVALID_CE(primop, op)
        }
        primop->unvisit(pass_);
        // the primop seems to be cycle free
        // => TODO: verify types
        return true;
    }

private:
    World& world_;
    InvalidEntries& invalid_;
    size_t pass_;
};

bool verify(World& world, InvalidEntries& invalid) {
    Verifier ver(world, invalid);
    return ver.verify();
}

// Invalid Entry class

InvalidEntry::InvalidEntry(InvalidEntryType type, const Def* def)
    : type_(type), def_(def), source_(def)
{
    assert( type != INVALID_CE );
}

InvalidEntry::InvalidEntry(const Def* def, const Def* source)
    : type_(INVALID_CE), def_(def), source_(source)
{
    assert( source != def );
}

void InvalidEntry::dump() {
    std::ostream& out = std::cout;
    out << "Invalid entry [";
    Printer::dump_name(out, def_, true);
    out << "]: ";
    switch(type_)
    {
    case INVALID_CE:
        out << "caused by ";
        Printer::dump_name(out, source_, true);
        break;
    case INVALID_NOT_IN_WORLD:
        out << "element is not in the current world";
        break;
    case INVALID_TYPES:
        out << "invalid types";
        break;
    case INVALID_TYPES_GENERICS:
        out << "cannot deduct invocation types";
        break;
    case INVALID_STRUCTURE:
        out << "invalid structure";
        break;
    case INVALID_CYCLIC_DEPENDENCY:
        out << "cyclic dependencies detected";
        break;
    default:
        out << "unknown cause";
        break;
    }
    out << std::endl;
    if(Lambda* lambda = def_->isa_lambda())
        lambda->dump(true, 0);
    else
        def_->dump(true);
    out << std::endl;
}


}
