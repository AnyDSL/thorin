#ifndef IMPALA_EMIT_H
#define IMPALA_EMIT_H

namespace anydsl2 {
    class World;
}

namespace impala {

class Prg;

void emit(anydsl2::World&, const Prg*);

} // namespace impala

#endif // IMPALA_EMIT_H
