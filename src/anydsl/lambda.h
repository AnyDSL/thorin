#ifndef ANYDSL_LAMBDA_H
#define ANYDSL_LAMBDA_H

#include <boost/container/flat_set.hpp>
#include <boost/unordered_set.hpp>

#include "anydsl/def.h"
#include "anydsl/util/autoptr.h"

namespace anydsl {

class Lambda;
class Pi;

typedef boost::unordered_set<const Lambda*> LambdaSet;
typedef ArrayRef<const Lambda*> Lambdas;

struct ParamLess {
    bool operator () (const Param* p1, const Param* p2) const {
        anydsl_assert(!p1->lambda() || !p2->lambda() || p1->lambda() == p2->lambda(), 
                "params belong to different lambdas"); 
        return p1->index() < p2->index(); 
    }
};

typedef boost::container::flat_set<const Param*, ParamLess> Params;

class Lambda : public Def {
public:

    enum Flags {
        Extern = 1 << 0,
    };

    Lambda(const Pi* pi, uint32_t flags = 0);
    virtual ~Lambda();
    virtual Lambda* clone() const { ANYDSL_UNREACHABLE; }

    const Param* appendParam(const Type* type);

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
    Lambdas succ()    const { return Lambdas(adjacencies_); }
    LambdaSet callers() const;
    const Params& params() const { return params_; }
    const Param* param(size_t i) const;
    Array<const Param*> copyParams() const;
    const Def* to() const { return op(0); };
    Ops args() const { return ops().slice_back(1); }
    const Def* arg(size_t i) const { return args()[i]; }
    const Pi* pi() const;
    const Pi* to_pi() const;
    uint32_t flags() const { return flags_; }

    void dump(bool fancy = false, int indent = 0) const;

    bool isExtern() const { return flags_ & Extern; }

    /**
     * @brief Removes the arguments specified in \p drop from the call.
     * The indices are specified in argument indices \em not in operand inidices,
     * i.e., arg_index = op_index + 1.
     */
    void shrink(ArrayRef<size_t> drop);

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
