#ifndef IMPALA_EMIT_H
#define IMPALA_EMIT_H

namespace anydsl2 {
class World;
}

namespace impala {

class Prg;

void emit(World& world, const Prg* prg);

} // namespace impala

#endif // IMPALA_EMIT_H
