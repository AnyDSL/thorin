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

    // higher order params
    Params::const_iterator ho_begin() const;
    Params::const_iterator ho_end() const { return params_.end(); }
    void ho_next(Params::const_iterator& i) const;
    bool is_higher_order() const { return ho_begin() != ho_end(); }

    // first order params
    Params::const_iterator fo_begin() const;
    Params::const_iterator fo_end() const { return params_.end(); }
    void fo_next(Params::const_iterator& i) const;
    bool is_first_order() const { return fo_begin() != fo_end(); }

    void close();

    Lambdas targets() const { return adjacencies_.slice_front(hosBegin_); }
    Lambdas hos()     const { return adjacencies_.slice_back(hosBegin_); }
    Lambdas succs()    const { return Lambdas(adjacencies_); }
    LambdaSet preds() const;
    const Params& params() const { return params_; }
    const Param* param(size_t i) const { 
        anydsl_assert(i < params_.size(), "index out of bounds"); 
        return params_[i]; 
    }
    const Def* to() const { return op(0); };
    Ops args() const { return ops().slice_back(1); }
    const Def* arg(size_t i) const { return args()[i]; }
    const Pi* pi() const;
    const Pi* to_pi() const;
    uint32_t flags() const { return flags_; }

    void dump(bool fancy = false, int indent = 0) const;

    bool is_extern() const { return flags_ & Extern; }

private:

    virtual bool equal(const Def* other) const;
    virtual size_t hash() const;
    virtual void vdump(Printer& printer) const;

    Params params_;
    uint32_t flags_;

    /// targets -- lambda arguments -- callers
    Array<const Lambda*> adjacencies_;
    size_t hosBegin_;

    friend class World;
    friend class Param;
};

} // namespace anydsl

#endif
