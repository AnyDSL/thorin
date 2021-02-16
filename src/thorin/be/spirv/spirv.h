#ifndef THORIN_SPIRV_H
#define THORIN_SPIRV_H

#include "thorin/be/backends.h"

namespace thorin::spirv {

class CodeGen : public thorin::CodeGen {
public:
    CodeGen(World&, Cont2Config&, bool debug);

    void emit(std::ostream& stream) override;
protected:
    void emit(const Scope& scope);
};

}

#endif //THORIN_SPIRV_H
