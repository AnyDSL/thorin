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

    virtual void index_out_of_range(uint64_t arity, uint64_t index);
    virtual void incomplete_match(const Match*);
    virtual void redundant_match_cases(const Match*, size_t index);
};

}

#endif
