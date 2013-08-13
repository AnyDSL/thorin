#ifndef IMPALA_SEMA_H
#define IMPALA_SEMA_H

namespace impala {

class TypeTable;

bool check(TypeTable&, const Prg*, bool nossa = false);

} // namespace impala

#endif // IMPALA_SEMA_H
