#ifndef ANYDSL2_JUMPTARGET_H
#define ANYDSL2_JUMPTARGET_H

namespace anydsl2 {

class Lambda;
class World;

class JumpTarget {
public:

    JumpTarget(const char* name = "")
        : lambda_(0)
        , first_(false)
        , name_(name)
    {}

    void target_by(Lambda* lambda);
    Lambda* enter();
    Lambda* enter_unsealed(World& world);
    World& world() const;

private:

    Lambda* lambda_;
    bool first_;
    const char* name_;

    friend class Lambda;
};

} // namespace anydsl2

#endif
