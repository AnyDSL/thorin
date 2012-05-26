#ifndef ANYDSL_JUMP_H
#define ANYDSL_JUMP_H

#include "anydsl/airnode.h"
#include "anydsl/lambda.h"
#include "anydsl/defuse.h"

namespace anydsl {

class NoRet;

class Jump : public Def {
public:

    Jump(Lambda* From, Def* to);

    Lambda* toLambda() { return ccast<Lambda>(to.def()->isa<Lambda>()); }
    const Lambda* toLambda() const { return to.def()->isa<Lambda>(); }

    Lambda* fromLambda() { return ccast<Lambda>(from.def()->as<Lambda>()); }
    const Lambda* fromLambda() const { return from.def()->as<Lambda>(); }

    const NoRet* noret() const;

    Ops& ops() { return ops_; }

    struct Args {
        typedef Ops::iterator iterator;
        typedef Ops::const_iterator const_iterator;
        typedef Ops::reverse_iterator reverse_iterator;
        typedef Ops::const_reverse_iterator const_reverse_iterator;

        Args(Jump& jump)
            : jump_(jump)
        {}

        iterator begin() { return jump_.args_begin_; }
        iterator end() { return jump_.ops_.end(); }
        const_iterator begin() const { return jump_.args_begin_; }
        const_iterator end() const { return jump_.ops_.end(); }

        reverse_iterator rbegin() { return jump_.ops_.rbegin(); }
        reverse_iterator rend() { return (--jump_.args_begin_).switch_direction(); }
        const_reverse_iterator rbegin() const { return jump_.ops_.rbegin(); }
        const_reverse_iterator rend() const { return (--jump_.args_begin_).switch_direction(); }

        Jump& jump_;
    };

    Args args() { return Args(*this); }

    const Use& from;///< May be a Lambda (in the case of a direct jump).
    const Use& to;  ///< Must be a Lambda.

private:

    Ops::iterator args_begin_;

    friend class World;
    friend class Args;
};

} // namespace anydsl

#endif
