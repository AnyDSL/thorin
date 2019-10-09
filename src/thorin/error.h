#ifndef THORIN_ERROR_H
#define THORIN_ERROR_H

#include <cstdint>

namespace thorin {

class Def;

class ErrorHandler {
public:
    virtual ~ErrorHandler() {};

    virtual void index_out_of_range(uint64_t arity, uint64_t index) = 0;
    virtual void empty_cases() = 0;
    virtual void match_cases_inconsistent(const Def* t1, const Def* t2) = 0;
};

class DefaultHandler : public ErrorHandler {
public:
    void index_out_of_range(uint64_t arity, uint64_t index) override;
    void empty_cases() override;
    void match_cases_inconsistent(const Def* t1, const Def* t2) override;
};

}

#endif
