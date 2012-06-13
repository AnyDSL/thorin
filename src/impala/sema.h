#ifndef IMPALA_SEMA_H
#define IMPALA_SEMA_H

namespace impala {

class TypeTable;

bool check(TypeTable& types, const Prg* prg);

} // namespace impala

#endif // IMPALA_SEMA_H
