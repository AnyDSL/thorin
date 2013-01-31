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

    World& world() const;
    void seal();
    void jump_from(Lambda* bb);

private:

    Lambda* get(World& world);
    Lambda* untangle();
    Lambda* enter();
    Lambda* enter_unsealed(World& world);

    Lambda* lambda_;
    bool first_;
    const char* name_;

    friend class IRBuilder;
};

//------------------------------------------------------------------------------

class IRBuilder {
public:

    IRBuilder(World& world)
        : cur_bb(0)
        , world_(world)
    {}

    World& world() const { return world_; }
    bool reachable() const { return cur_bb; }
    Lambda* enter(JumpTarget& jt) { return cur_bb = jt.enter(); }
    Lambda* enter_unsealed(JumpTarget& jt) { return cur_bb = jt.enter_unsealed(world_); }
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
