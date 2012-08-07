#ifdef ANYDSL_UF_TYPE
#   define ANYDSL_U_TYPE(T) ANYDSL_UF_TYPE(T)
#   define ANYDSL_F_TYPE(T) ANYDSL_UF_TYPE(T)
#elif defined ANYDSL_JUST_U_TYPE
#   define ANYDSL_U_TYPE(T) ANYDSL_JUST_U_TYPE(T)
#   define ANYDSL_F_TYPE(foo)
#elif defined ANYDSL_JUST_F_TYPE
#   define ANYDSL_U_TYPE(foo)
#   define ANYDSL_F_TYPE(T) ANYDSL_JUST_F_TYPE(T)
#else
#   ifndef ANYDSL_U_TYPE
#       error "define ANYDSL_U_TYPE before including this file"
#   endif
#   ifndef ANYDSL_F_TYPE
#       error "define ANYDSL_F_TYPE before including this file"
#   endif
#endif

ANYDSL_U_TYPE(u1)
ANYDSL_U_TYPE(u8)
ANYDSL_U_TYPE(u16)
ANYDSL_U_TYPE(u32)
ANYDSL_U_TYPE(u64)

ANYDSL_F_TYPE(f32)
ANYDSL_F_TYPE(f64)

#undef ANYDSL_U_TYPE
#undef ANYDSL_F_TYPE
#undef ANYDSL_UF_TYPE
#undef ANYDSL_JUST_U_TYPE
#undef ANYDSL_JUST_F_TYPE
