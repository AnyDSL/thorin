#ifndef THORIN_BE_AIR_H
#define THORIN_BE_AIR_H

namespace thorin {

class DefNode;
class Lambda;
class PrimOp;
class Type;
class World;

void emit_thorin(World&, bool fancy = false, bool colored = true);
void emit_type(const Type*);
void emit_def(Def);
void emit_assignment(const PrimOp*);
void emit_head(const Lambda*);
void emit_jump(const Lambda*);

}

#endif
