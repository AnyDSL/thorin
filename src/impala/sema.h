#ifndef IMPALA_SEMA_H
#define IMPALA_SEMA_H

namespace impala {

class Scope;
class TypeTable;

bool check(TypeTable&, const Scope*, bool nossa = false);

} // namespace impala

#endif // IMPALA_SEMA_H
