#ifndef IMPALA_SEMA_H
#define IMPALA_SEMA_H

namespace impala {

class World;

bool check(World& world, const Prg* prg, bool nossa = false);

} // namespace impala

#endif // IMPALA_SEMA_H
