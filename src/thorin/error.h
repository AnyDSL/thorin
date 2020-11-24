#ifndef THORIN_ERROR_H
#define THORIN_ERROR_H

#include <cstdint>
#include <cstddef>

namespace thorin {

class Def;
class Ptrn;
class Match;

class ErrorHandler {
public:
    virtual ~ErrorHandler() {};

    virtual void index_out_of_range(const Def* arity, const Def* index);
    virtual void ill_typed_app(const Def* callee, const Def* arg);
    virtual void incomplete_match(const Match*);
    virtual void redundant_match_case(const Match*, const Ptrn*);
};

}

#endif
