/*
 * !!! DO NOT CHANGE THIS ORDER !!!
 */

/*
 * prefix operators
 */

#ifndef IMPALA_PREFIX
#define IMPALA_PREFIX(tok, str, prec)
#endif

//              token type, string, left prec,  right prec
IMPALA_PREFIX(         ADD,    "+",   POSTFIX             ) // unary +
IMPALA_PREFIX(         SUB,    "-",   POSTFIX             ) // unary -
IMPALA_PREFIX(         MUL,    "*",   POSTFIX             ) // deref
IMPALA_PREFIX(         AND,    "&",   POSTFIX             ) // address of
IMPALA_PREFIX(         NOT,    "~",   POSTFIX             ) // unary complement
IMPALA_PREFIX(         L_N,    "!",   POSTFIX             ) // logical not
IMPALA_PREFIX(         INC,   "++",   POSTFIX             ) // prefix ++
IMPALA_PREFIX(         DEC,   "--",   POSTFIX             ) // prefix --
IMPALA_PREFIX(          OR,    "|",   POSTFIX             ) // lambda expressions
IMPALA_PREFIX(         L_O,   "||",   POSTFIX             ) // lambda expressions with empty param list
IMPALA_PREFIX(         RUN,    "@",   POSTFIX             ) // trigger partial evaluation
IMPALA_PREFIX(        HALT,    "$",   POSTFIX             ) // stop partial evaluation

#undef IMPALA_PREFIX

/*
 * postfix operators
 */

#ifndef IMPALA_POSTFIX
#define IMPALA_POSTFIX(tok, str, prec)
#endif

IMPALA_POSTFIX(        INC,   "++",                POSTFIX) // postfix ++
IMPALA_POSTFIX(        DEC,   "--",                POSTFIX) // postfix --
IMPALA_POSTFIX(  L_BRACKET,    "[",                POSTFIX) // static args
IMPALA_POSTFIX(    L_PAREN,    "(",                POSTFIX) // function call

#undef IMPALA_POSTFIX

/*
 * binary operators
 */

#ifndef IMPALA_INFIX_ASGN
#define IMPALA_INFIX_ASGN(tok, str, lprec, rprec)
#endif

IMPALA_INFIX_ASGN(    ASGN,    "=",      COND,        ASGN)
IMPALA_INFIX_ASGN(ADD_ASGN,   "+=",      COND,        ASGN)
IMPALA_INFIX_ASGN(SUB_ASGN,   "-=",      COND,        ASGN)
IMPALA_INFIX_ASGN(MUL_ASGN,   "*=",      COND,        ASGN)
IMPALA_INFIX_ASGN(DIV_ASGN,   "/=",      COND,        ASGN)
IMPALA_INFIX_ASGN(MOD_ASGN,   "%=",      COND,        ASGN)
IMPALA_INFIX_ASGN(AND_ASGN,   "&=",      COND,        ASGN)
IMPALA_INFIX_ASGN( OR_ASGN,   "|=",      COND,        ASGN)
IMPALA_INFIX_ASGN(XOR_ASGN,   "^=",      COND,        ASGN)
IMPALA_INFIX_ASGN(SHL_ASGN,  "<<=",      COND,        ASGN)
IMPALA_INFIX_ASGN(SHR_ASGN,  ">>=",      COND,        ASGN)

#undef IMPALA_INFIX_ASGN

#ifndef IMPALA_INFIX
#define IMPALA_INFIX(tok, str, lprec, rprec)
#endif

IMPALA_INFIX(QUESTION_MARK,    "?",       L_O,        ASGN)

IMPALA_INFIX(          L_O,   "||",       L_O,         L_A)
IMPALA_INFIX(          L_A,   "&&",       L_A,          OR)

IMPALA_INFIX(           OR,    "|",        OR,         XOR)
IMPALA_INFIX(          XOR,    "^",       XOR,         AND)
IMPALA_INFIX(          AND,    "&",       AND,          EQ)

IMPALA_INFIX(           EQ,   "==",        EQ,         REL)
IMPALA_INFIX(           NE,   "!=",        EQ,         REL)

IMPALA_INFIX(           LT,    "<",       REL,       SHIFT)
IMPALA_INFIX(           LE,   "<=",       REL,       SHIFT)
IMPALA_INFIX(           GT,    ">",       REL,       SHIFT)
IMPALA_INFIX(           GE,   ">=",       REL,       SHIFT)

IMPALA_INFIX(          SHL,   "<<",     SHIFT,         ADD)
IMPALA_INFIX(          SHR,   ">>",     SHIFT,         ADD)

IMPALA_INFIX(          ADD,   "+",        ADD,         MUL)
IMPALA_INFIX(          SUB,   "-",        ADD,         MUL)

IMPALA_INFIX(          MUL,   "*",        MUL,       UNARY)
IMPALA_INFIX(          DIV,   "/",        MUL,       UNARY)
IMPALA_INFIX(          MOD,   "%",        MUL,       UNARY)

IMPALA_INFIX(          DOT,   ".",    POSTFIX,         TOP)
IMPALA_INFIX(        ARROW,  "->",    POSTFIX,         TOP)

#undef IMPALA_INFIX

/*
 * keywords
 */

#ifndef IMPALA_KEY_EXPR
#define IMPALA_KEY_EXPR(tok, str)
#endif

IMPALA_KEY_EXPR(TRUE,   "true")
IMPALA_KEY_EXPR(FALSE,  "false")

#undef IMPALA_KEY_EXPR

#ifndef IMPALA_KEY_STMT
#define IMPALA_KEY_STMT(tok, str)
#endif

IMPALA_KEY_STMT(IF,         "if")
IMPALA_KEY_STMT(ELSE,       "else")
IMPALA_KEY_STMT(WHILE,      "while")
IMPALA_KEY_STMT(DO,         "do")
IMPALA_KEY_STMT(FOR,        "for")
IMPALA_KEY_STMT(FOREACH,    "foreach")
IMPALA_KEY_STMT(BREAK,      "break")
IMPALA_KEY_STMT(CONTINUE,   "continue")
IMPALA_KEY_STMT(RETURN,     "return")
IMPALA_KEY_STMT(LET,        "let")

#undef IMPALA_KEY_STMT

#ifndef IMPALA_KEY_MISC
#define IMPALA_KEY_MISC(tok, str)
#endif

IMPALA_KEY_MISC(EXTERN,     "extern")
IMPALA_KEY_MISC(INTRINSIC,  "intrinsic")
IMPALA_KEY_MISC(FN,         "fn")
IMPALA_KEY_MISC(TYPE_int,   "int")
IMPALA_KEY_MISC(TYPE_noret, "noret")
IMPALA_KEY_MISC(TYPE_uint,  "uint")
IMPALA_KEY_MISC(TYPE_void,  "void")
IMPALA_KEY_MISC(STRUCT,     "struct")
IMPALA_KEY_MISC(TRAIT,      "trait")

#undef IMPALA_KEY_MISC

/*
 * other miscellaneous tokens
 */

#ifndef IMPALA_MISC
#define IMPALA_MISC(tok, str)
#endif

IMPALA_MISC(L_PAREN,    "(")
IMPALA_MISC(R_PAREN,    ")")
IMPALA_MISC(L_BRACE,    "{")
IMPALA_MISC(R_BRACE,    "}")
IMPALA_MISC(L_BRACKET,  "[")
IMPALA_MISC(R_BRACKET,  "]")
IMPALA_MISC(COMMA,      ",")
IMPALA_MISC(SEMICOLON,  ";")
IMPALA_MISC(COLON,      ":")
IMPALA_MISC(LARROW,     "<-")

#undef IMPALA_MISC

/*
 * literals
 */

#ifndef IMPALA_LIT
#define IMPALA_LIT(itype, atype)
#endif

IMPALA_LIT(int8,    ps8)
IMPALA_LIT(int16,  ps16)
IMPALA_LIT(int32,  ps32)
IMPALA_LIT(int64,  ps64)
IMPALA_LIT(float,  pf32)
IMPALA_LIT(double, pf64)

#undef IMPALA_LIT

#ifndef IMPALA_TYPE
#define IMPALA_TYPE(itype, atype)
#endif

IMPALA_TYPE(  int8,  ps8)
IMPALA_TYPE( int16, ps16)
IMPALA_TYPE( int32, ps32)
IMPALA_TYPE( int64, ps64)
IMPALA_TYPE( float, pf32)
IMPALA_TYPE(double, pf64)
IMPALA_TYPE(  bool, bool)

#undef IMPALA_TYPE
