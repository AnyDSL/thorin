#ifndef ANYDSL_LAMBDA_H
#define ANYDSL_LAMBDA_H

#include <vector>
#include <boost/unordered_set.hpp>

#include "anydsl/def.h"
#include "anydsl/util/autoptr.h"

namespace anydsl {

class Lambda;
class Pi;

typedef boost::unordered_set<const Lambda*> LambdaSet;

typedef std::vector<const Param*> Params;

class Lambda : public Def {
public:

    enum Flags {
        Extern = 1 << 0,
    };

    Lambda(const Pi* pi, uint32_t flags = 0);
    virtual Lambda* clone() const { ANYDSL_UNREACHABLE; }

    const Param* append_param(const Type* type);

    Array<const Param*> fo_params() const;
    Array<const Param*> ho_params() const;
    Array<const Def*> fo_args() const;
    Array<const Def*> ho_args() const;
    bool is_fo() const;
    bool is_ho() const;

    void close(size_t gid);

    typedef ArrayRef<const Lambda*> Lambdas;

    Lambdas targets() const { return adjacencies_.slice_front(hos_begin_); }
    Lambdas hos()     const { return adjacencies_.slice_back(hos_begin_); }
    Lambdas succs()    const { return Lambdas(adjacencies_); }
    LambdaSet preds() const;
    const Params& params() const { return params_; }
    const Param* param(size_t i) const { return params_[i]; }
    const Def* to() const { return op(0); };
    Ops args() const { return ops().slice_back(1); }
    const Def* arg(size_t i) const { return args()[i]; }
    const Pi* pi() const;
    const Pi* to_pi() const;
    uint32_t flags() const { return flags_; }
    size_t gid() const { return gid_; }
    size_t sid() const { return sid_; }

    /**
     * Is this Lambda part of a call-lambda-cascade? <br>
     * @code
lambda(...) jump (foo, [..., lambda(...) ..., ...]
     * @endcode
     */
    bool is_cascading() const;

    void dump(bool fancy = false, int indent = 0) const;

    bool is_extern() const { return flags_ & Extern; }

    bool sid_valid() const { return sid_ != size_t(-1); }
    bool sid_invalid() const { return sid_ == size_t(-1); }
    void invalidate_sid() const { sid_ = size_t(-1); }

private:

    template<bool fo> Array<const Param*> classify_params() const;
    template<bool fo> Array<const Def*> classify_args() const;

    virtual bool equal(const Def* other) const;
    virtual size_t hash() const;
    virtual void vdump(Printer& printer) const;

    Params params_;
    uint32_t flags_;

    /// targets -- lambda arguments -- callers
    Array<const Lambda*> adjacencies_;
    size_t hos_begin_;
    size_t gid_;        ///< global index
    mutable size_t sid_; ///< scope index

    friend class World;
    friend class Param;
    friend size_t number(const LambdaSet& lambdas, const Lambda* cur, size_t i);
};

} // namespace anydsl

#endif
