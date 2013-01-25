#ifndef ANYDSL2_JUMPTARGET_H
#define ANYDSL2_JUMPTARGET_H

namespace anydsl2 {

class Def;
class Lambda;
class World;

class JumpTarget {
public:

    JumpTarget(const char* name = "")
        : lambda_(0)
        , first_(false)
        , name_(name)
    {}

    Lambda* enter();
    Lambda* enter_unsealed(World& world);
    World& world() const;
    void seal();
    void jump(JumpTarget& to);
    void branch(const Def* cond, JumpTarget& tto, JumpTarget& fto);

private:

    void untangle_first();
    void new_lambda(World& world);

    Lambda* lambda_;
    bool first_;
    const char* name_;

    friend class Lambda;
};

} // namespace anydsl2

#endif
