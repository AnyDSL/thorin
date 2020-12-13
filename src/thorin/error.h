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

    virtual void expected_shape(const Def* def);
    virtual void index_out_of_range(const Def* arity, const Def* index);
    virtual void ill_typed_app(const Def* callee, const Def* arg);
};

}

#endif
