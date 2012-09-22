#ifndef ANYDSL_LAMBDA_H
#define ANYDSL_LAMBDA_H

#include <vector>
#include <boost/unordered_set.hpp>

#include "anydsl/def.h"
#include "anydsl/util/autoptr.h"

namespace anydsl {

class Lambda;
class Pi;

typedef boost::unordered_set<Lambda*> LambdaSet;

typedef std::vector<const Param*> Params;

class Lambda : public Def {
private:

    Lambda(size_t gid, const Pi* pi, uint32_t flags);
    virtual ~Lambda();

public:

    enum Flags {
        Extern = 1 << 0,
    };

    virtual Lambda* clone() const { ANYDSL_UNREACHABLE; }
    Lambda* stub() const;

    const Param* append_param(const Type* type);

    Array<const Param*> fo_params() const;
    Array<const Param*> ho_params() const;
    Array<const Def*> fo_args() const;
    Array<const Def*> ho_args() const;
    bool is_fo() const;
    bool is_ho() const;

    LambdaSet targets() const;
    LambdaSet hos() const;
    LambdaSet succs() const;
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

    bool sid_valid() { return sid_ != size_t(-1); }
    bool sid_invalid() { return sid_ == size_t(-1); }
    void invalidate_sid() { sid_ = size_t(-1); }

    void jump(const Def* to, ArrayRef<const Def*> args);
    void jump0(const Def* to) {
        return jump(to, ArrayRef<const Def*>(0, 0));
    }
    void jump1(const Def* to, const Def* arg1) {
        const Def* args[1] = { arg1 };
        return jump(to, args);
    }
    void jump2(const Def* to, const Def* arg1, const Def* arg2) {
        const Def* args[2] = { arg1, arg2 };
        return jump(to, args);
    }
    void jump3(const Def* to, const Def* arg1, const Def* arg2, const Def* arg3) {
        const Def* args[3] = { arg1, arg2, arg3 };
        return jump(to, args);
    }
    void branch(const Def* cond, const Def* tto, const Def* fto);

    Lambda* drop(size_t i, const Def* with);
    Lambda* drop(ArrayRef<size_t> args, ArrayRef<const Def*> with);
    Lambda* drop(ArrayRef<const Def*> with);

private:

    template<bool fo> Array<const Param*> classify_params() const;
    template<bool fo> Array<const Def*> classify_args() const;

    virtual bool equal(const Def* other) const;
    virtual size_t hash() const;
    virtual void vdump(Printer& printer) const;

    size_t gid_; ///< global index
    uint32_t flags_;
    Params params_;
    size_t sid_; ///< scope index

    friend class World;
    friend class Param;
    friend size_t number(const LambdaSet& lambdas, Lambda* cur, size_t i);
};

} // namespace anydsl

#endif
