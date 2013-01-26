#ifndef ANYDSL2_JUMPTARGET_H
#define ANYDSL2_JUMPTARGET_H

#include "anydsl2/util/array.h"

namespace anydsl2 {

class Def;
class Lambda;
class Type;
class World;

//------------------------------------------------------------------------------

class JumpTarget {
public:

    JumpTarget(const char* name = "")
        : lambda_(0)
        , first_(false)
        , name_(name)
    {}
#ifndef NDEBUG
    ~JumpTarget();
#endif

    Lambda* enter();
    Lambda* enter_unsealed(World& world);
    World& world() const;
    void seal();

private:

    Lambda* get(World& world);
    void untangle_first();
    Lambda* new_lambda(World& world);

    Lambda* lambda_;
    bool first_;
    const char* name_;

    friend class Builder;
};

//------------------------------------------------------------------------------

class Builder {
public:

    Builder(World& world)
        : cur_bb(0)
        , world_(world)
    {}

    World& world() const { return world_; }
    bool reachable() const { return cur_bb; }
    void enter(JumpTarget& jt) { jump(jt); cur_bb = jt.enter(); }
    void enter_unsealed(JumpTarget& jt) { jump(jt); cur_bb = jt.enter_unsealed(world_); }
    void jump(JumpTarget& jt);
    void branch(const Def* cond, JumpTarget& t, JumpTarget& f);
    void call(const Def* to, ArrayRef<const Def*> args, const Type* ret_type);

    Lambda* cur_bb;

protected:

    World& world_;
};

//------------------------------------------------------------------------------

} // namespace anydsl2

#endif
