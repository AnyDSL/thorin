#ifndef ANYDSL_JUMP_H
#define ANYDSL_JUMP_H

#include "anydsl/airnode.h"
#include "anydsl/lambda.h"
#include "anydsl/def.h"

namespace anydsl {

class NoRet;

class Jump : public Def {
protected:

    Jump(World& world, IndexKind indexKind, size_t numOps);

public:

    struct Args {
        typedef const Def** const_iterator;
        typedef std::reverse_iterator<const Def**> const_reverse_iterator;

        Args(const Jump& jump, size_t begin, size_t end) 
            : jump_(jump) 
            , begin_(begin)
            , end_(end)
        {}

        const_iterator begin() const { return jump_.ops_ + begin_; }
        const_iterator end() const { return jump_.ops_ + end_; }
        const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
        const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

        const Def* const& operator [] (size_t i) const {
            anydsl_assert(i < size(), "index out of bounds");
            return jump_.ops_[begin_ + i];
        }

        size_t size() const { return end_ - begin_; }
        bool empty() const { return begin_ == end_; }

        const Def*& front() const { assert(!empty()); return *(jump_.ops_ + begin_); }
        const Def*& back()  const { assert(!empty()); return *(jump_.ops_ + end_ - 1); }

    private:

        const Jump& jump_;
        size_t begin_;
        size_t end_;
    };

    const NoRet* noret() const;
};

//------------------------------------------------------------------------------

class Goto : public Jump {
private:

    Goto(World& world, const Def* to, const Def* const* begin, const Def* const* end);

public:

    const Lambda* lambda() const { return to()->isa<Lambda>(); }

    Args args() const { return Args(*this, 1, numOps()); }

    virtual void dump(Printer& printer, LambdaPrinterMode mode) const;

    const Def* to() const { return ops_[0]; };

private:

    friend class World;
    friend class Args;
};
//------------------------------------------------------------------------------

class Branch : public Jump {
private:

    Branch(World& world, const Def* cond, 
            const Def* tto, const Def* const* tbegin, const Def* const* tend,
            const Def* fto, const Def* const* fbegin, const Def* const* fend);

public:

    const Lambda* tLambda() const { return tto()->isa<Lambda>(); }
    const Lambda* fLambda() const { return fto()->isa<Lambda>(); }

    const Def* cond() const { return ops_[0]; };
    const Def* tto() const { return ops_[1]; };
    const Def* fto() const { return ops_[findex_]; };

    Args targs() const { return Args(*this, 2, findex_); }
    Args fargs() const { return Args(*this, findex_ + 1, numOps()); }

    virtual void dump(Printer& printer, LambdaPrinterMode mode) const;

private:

    size_t findex_;

    friend class World;
    friend class Args;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif
