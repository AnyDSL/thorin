#ifndef ANYDSL_JUMP_H
#define ANYDSL_JUMP_H

#include "anydsl/airnode.h"
#include "anydsl/lambda.h"
#include "anydsl/defuse.h"

namespace anydsl {

class NoRet;

class Jump : public Value {
private:

    Jump(Def* to, Def* const* begin, Def* const* end);

public:

    Lambda* toLambda() { return to().def()->isa<Lambda>(); }
    const Lambda* toLambda() const { return to().def()->isa<Lambda>(); }

    const NoRet* noret() const;

    struct Args {
        typedef Use* iterator;
        typedef Use* const_iterator;
        typedef std::reverse_iterator<Use*> reverse_iterator;
        typedef std::reverse_iterator<Use*> const_reverse_iterator;

        Args(Jump& jump) : jump(jump) {}

        iterator begin() { return jump.ops_ + 1; }
        iterator end() { return jump.ops_ + size(); }
        const_iterator begin() const { return jump.ops_ + 1; }
        const_iterator end() const { return jump.ops_ + size(); }

        reverse_iterator rbegin() { return reverse_iterator(jump.ops_ + size()); }
        reverse_iterator rend() { return reverse_iterator(jump.ops_ + 1); }
        const_reverse_iterator rbegin() const { return reverse_iterator(jump.ops_ + size()); }
        const_reverse_iterator rend() const { return reverse_iterator(jump.ops_ + 1); }

        size_t size() const { return jump.numOps() - 1; }
        bool empty() const { return jump.numOps() == 1; }

        Jump& jump;
    };

    Args args() { return Args(*this); }

    Use& to() { return ops_[0]; }
    const Use& to() const { return ops_[0]; };

private:

    friend class World;
    friend class Args;
};

} // namespace anydsl

#endif
