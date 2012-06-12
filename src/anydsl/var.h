#ifndef ANYDSL_VAR_H
#define ANYDSL_VAR_H

namespace anydsl {

class Var {
public:

    Var(const Def* def)
        : def_(def)
    {}
    virtual ~Var() {}

    const Def* load() const { return def_; }
    virtual void store(const Def* def) = 0;
    const Type* type() const { return def_->type(); }

private:

    const Def* def_;
};

class RVar : public Var {
public:

    LVar();

    virtual void store(const Def* def) { assert(false); }

private:

    anydsl::Symbol sym_;
};

class LVar : public Var {
public:

    LVar();

    virtual void store(const Def* def) { def_ = def; }

private:

    anydsl::Symbol sym_;
};

} // namespace anydsl

#endif // ANYDSL_VAR_H
