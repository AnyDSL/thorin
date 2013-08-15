#ifndef ANYDSL2_BE_AIR_H
#define ANYDSL2_BE_AIR_H

namespace anydsl2 {

class Def;
class Lambda;
class PrimOp;
class Type;
class World;

void emit_air(World&, bool fancy = false);
void emit_type(const Type*);
void emit_def(const Def*);
void emit_assignment(const PrimOp*);
void emit_head(const Lambda*);
void emit_jump(const Lambda*);

}

#endif
