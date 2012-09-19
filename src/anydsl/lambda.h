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
typedef ArrayRef<const Lambda*> Lambdas;

typedef std::vector<const Param*> Params;

class Lambda : public Def {
public:

    enum Flags {
        Extern = 1 << 0,
    };

    Lambda(const Pi* pi, uint32_t flags = 0);
    virtual Lambda* clone() const { ANYDSL_UNREACHABLE; }

    const Param* append_param(const Type* type);

    Array<const Param*> first_order_params() const;
    Array<const Param*> higher_order_params() const;
    Array<const Def*> first_order_args() const;
    Array<const Def*> higher_order_args() const;
    bool is_first_order() const;
    bool is_higher_order() const;

    void close(size_t gid);

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

    void dump(bool fancy = false, int indent = 0) const;

    bool is_extern() const { return flags_ & Extern; }

    mutable size_t lid; ///< local index
    bool lid_valid() const { return lid != size_t(-1); }
    bool lid_invalid() const { return lid == size_t(-1); }
    void invalidate_lid() const { lid = size_t(-1); }

private:

    template<bool first_order> Array<const Param*> classify_params() const;
    template<bool first_order> Array<const Def*> classify_args() const;

    virtual bool equal(const Def* other) const;
    virtual size_t hash() const;
    virtual void vdump(Printer& printer) const;

    Params params_;
    uint32_t flags_;

    /// targets -- lambda arguments -- callers
    Array<const Lambda*> adjacencies_;
    size_t hos_begin_;
    size_t gid_; ///< global index

    friend class World;
    friend class Param;
};

} // namespace anydsl

#endif
