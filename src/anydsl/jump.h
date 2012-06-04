#ifndef ANYDSL_JUMP_H
#define ANYDSL_JUMP_H

#include "anydsl/airnode.h"
#include "anydsl/lambda.h"
#include "anydsl/defuse.h"

namespace anydsl {

class NoRet;

class Jump : public Value {
private:

    template<class T>
    Jump(Def* to, T begin, T end) 
        : Value(Index_Jump, 
                0 /*to->world().noret(to->type()->as<Pi>())*/, std::distance(begin, end) + 1)
    { 
        setOp(0, to);

        T i = begin;
        for (size_t x = 1; i != end; ++x, ++i)
            setOp(x, *i);
    }

public:

    Lambda* toLambda() { return to().def()->isa<Lambda>(); }
    const Lambda* toLambda() const { return to().def()->isa<Lambda>(); }

    const NoRet* noret() const;

#if 0
    Ops& ops() { return ops_; }

    struct Args {
        typedef Use* iterator;
        typedef Use* const const_iterator;
        //typedef Ops::reverse_iterator reverse_iterator;
        //typedef Ops::const_reverse_iterator const_reverse_iterator;

        Args(Jump& jump)
            : jump_(jump)
        {}

        iterator begin() { return jump_.args_begin_; }
        iterator end() { return jump_.ops_.end(); }
        const_iterator begin() const { return jump_.args_begin_; }
        const_iterator end() const { return jump_.ops_.end(); }

#if 0
        reverse_iterator rbegin() { return jump_.ops_.rbegin(); }
        reverse_iterator rend() { return (--jump_.args_begin_).switch_direction(); }
        const_reverse_iterator rbegin() const { return jump_.ops_.rbegin(); }
        const_reverse_iterator rend() const { return (--jump_.args_begin_).switch_direction(); }
#endif

        size_t size() const { return jump_.ops_.size() - 1; }
        bool empty() const { return jump_.ops_.size() == 1; }

        Jump& jump_;
    };

    Args args() { return Args(*this); }
#endif

    Use& to() { return ops_[0]; }
    const Use& to() const { return ops_[0]; };

private:

    Ops::iterator args_begin_;

    friend class World;
    friend class Args;
};

} // namespace anydsl

#endif
