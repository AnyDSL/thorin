#ifndef THORIN_LAMBDA_H
#define THORIN_LAMBDA_H

#include <vector>

#include "thorin/def.h"
#include "thorin/type.h"
#include "thorin/util/autoptr.h"

namespace thorin {

class Lambda;

typedef std::vector<Lambda*> Lambdas;

//------------------------------------------------------------------------------

enum class Intrinsic : uint8_t {
    None,       ///< Not an intrinsic.
    CUDA,       ///< Internal CUDA-Backend.
    NVVM,       ///< Internal NNVM-Backend.
    SPIR,       ///< Internal SPIR-Backend.
    OpenCL,     ///< Internal OpenCL-Backend.
    Parallel,   ///< Internal Parallel-CPU-Backend.
    Vectorize,  ///< External vectorizer.
    Mmap,       ///< Intrinsic memory-mapping function.
    Munmap,     ///< Intrinsic memory-unmapping function.
};

class Lambda : public DefNode {
public:
    enum AttrKind {
        Extern       = 1 <<  0, ///< Is the function visible in other translation units?
        Device       = 1 <<  1, ///< Flag for intrinsic function with device calling convention.
        KernelEntry  = 1 <<  2, ///< Flag for the kernel lambda.
    };

    struct Attribute {
        explicit Attribute(uint32_t flags)
            : flags_(flags)
        {}

        uint32_t filter(uint32_t flags) const { return flags_ & flags; }
        bool is(uint32_t flags) const { return filter(flags) != 0; }
        void set(uint32_t flags) { flags_ |=  flags; }
        void clear(uint32_t flags = uint32_t(-1)) { flags_ &= ~flags; }
        void toggle(uint32_t flags) { flags_ ^= flags; }
        uint32_t flags() const { return flags_; }

    private:
        uint32_t flags_;
    };

private:
    Lambda(size_t gid, FnType fn, Attribute attribute, Intrinsic intrinsic, bool is_sealed, const std::string& name)
        : DefNode(gid, Node_Lambda, 0, fn, true, name)
        , attribute_(attribute)
        , intrinsic_(intrinsic)
        , parent_(this)
        , is_sealed_(is_sealed)
        , is_visited_(false)
    {
        params_.reserve(fn->num_args());
    }
    virtual ~Lambda() { for (auto param : params()) delete param; }

public:
    Lambda* stub() const { Type2Type map; return stub(map); }
    Lambda* stub(const std::string& name) const { Type2Type map; return stub(map, name); }
    Lambda* stub(Type2Type& type2type) const { return stub(type2type, name); }
    Lambda* stub(Type2Type& type2type, const std::string& name) const;
    Lambda* update_to(Def def) { return update_op(0, def); }
    Lambda* update_op(size_t i, Def def);
    Lambda* update_arg(size_t i, Def def) { return update_op(i+1, def); }
    const Param* append_param(Type type, const std::string& name = "");
    Lambdas direct_preds() const;
    Lambdas direct_succs() const;
    Lambdas indirect_preds() const;
    Lambdas indirect_succs() const;
    Lambdas preds() const;
    Lambdas succs() const;
    ArrayRef<const Param*> params() const { return params_; }
    Array<Def> params_as_defs() const;
    const Param* param(size_t i) const { assert(i < num_params()); return params_[i]; }
    const Param* mem_param() const;
    Def to() const { return op(0); };
    ArrayRef<Def> args() const { return empty() ? ArrayRef<Def>(0, 0) : ops().slice_from_begin(1); }
    Def arg(size_t i) const { return args()[i]; }
    FnType type() const { return DefNode::type().as<FnType>(); }
    FnType to_fn_type() const { return to()->type().as<FnType>(); }
    FnType arg_fn_type() const;
    size_t num_args() const { return args().size(); }
    size_t num_params() const { return params().size(); }
    Attribute& attribute() { return attribute_; }
    const Attribute& attribute() const { return attribute_; }
    Intrinsic& intrinsic() { return intrinsic_; }
    Intrinsic intrinsic() const { return intrinsic_; }
    void set_intrinsic(); ///< Sets \p intrinsic_ derived on this \p Lambda's \p name.
    /**
     * Is this Lambda part of a call-lambda-cascade? <br>
     * @code
lambda(...) jump (foo, [..., lambda(...) ..., ...]
     * @endcode
     */
    bool is_cascading() const;
    bool is_basicblock() const;
    bool is_returning() const;
    bool is_intrinsic() const;
    bool visit_connected_intrinsics(std::function<bool(Lambda*)> func) const;
    bool is_connected_to_intrinsic() const { return visit_connected_intrinsics([&] (Lambda*) { return true; }); }
    bool is_connected_to_intrinsic(Intrinsic intrinsic) const {
        return visit_connected_intrinsics([&] (Lambda* lambda) { return lambda->intrinsic() == intrinsic; });
    }
    void dump_head() const;
    void dump_jump() const;
    void destroy_body();

    // terminate

    void jump(Def to, ArrayRef<Def> args);
    void branch(Def cond, Def tto, Def fto, ArrayRef<Def> args = ArrayRef<Def>(nullptr, 0));
    std::pair<Lambda*, Def> call(Def to, ArrayRef<Def> args, Type ret_type);

    // cps construction

    Def set_value(size_t handle, Def def);
    Def get_value(size_t handle, Type type, const char* name = "");
    Def set_mem(Def def);
    Def get_mem();
    Lambda* parent() const { return parent_; }            ///< See \ref parent_ for more information.
    void set_parent(Lambda* parent) { parent_ = parent; } ///< See \ref parent_ for more information.
    void seal();
    bool is_sealed() const { return is_sealed_; }
    void unseal() { is_sealed_ = false; }
    void clear() { values_.clear(); }
    bool is_cleared() { return values_.empty(); }


private:
    class Todo {
    public:
        Todo() {}
        Todo(size_t handle, size_t index, Type type, const char* name)
            : handle_(handle)
            , index_(index)
            , type_(type)
            , name_(name)
        {}

        size_t handle() const { return handle_; }
        size_t index() const { return index_; }
        Type type() const { return type_; }
        const char* name() const { return name_; }

    private:
        size_t handle_;
        size_t index_;
        Type type_;
        const char* name_;
    };

    Def fix(size_t handle, size_t index, Type type, const char* name);
    Def try_remove_trivial_param(const Param*);
    Def find_def(size_t handle);
    void increase_values(size_t handle) { if (handle >= values_.size()) values_.resize(handle+1); }

    Attribute attribute_;
    Intrinsic intrinsic_;
    std::vector<const Param*> params_;
    /**
     * There exist three cases to distinguish here.
     * - \p parent_ == this: This \p Lambda is considered as a basic block, i.e.,
     *                       SSA construction will propagate value through this \p Lambda's predecessors.
     * - \p parent_ == nullptr: This \p Lambda is considered as top level function, i.e.,
     *                          SSA construction will stop propagate values here.
     *                          Any \p get_value which arrives here without finding a definition will return \p bottom.
     * - otherwise: This \p Lambda is considered as function head nested in \p parent_.
     *              Any \p get_value which arrives here without finding a definition will recursively try to find one in \p parent_.
     */
    Lambda* parent_;
    bool is_sealed_;
    bool is_visited_;
    std::vector<Def> values_;
    std::vector<Todo> todos_;

    friend class Cleaner;
    friend class World;
};

//------------------------------------------------------------------------------

template<class To>
using LambdaMap     = HashMap<Lambda*, To, GIDHash<Lambda*>, GIDEq<Lambda*>>;
using LambdaSet     = HashSet<Lambda*, GIDHash<Lambda*>, GIDEq<Lambda*>>;
using Lambda2Lambda = LambdaMap<Lambda*>;

//------------------------------------------------------------------------------

} // namespace thorin

#endif
