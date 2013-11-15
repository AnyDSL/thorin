#ifdef ANYDSL2_UF_TYPE
#   define ANYDSL2_U_TYPE(T) ANYDSL2_UF_TYPE(T)
#   define ANYDSL2_F_TYPE(T) ANYDSL2_UF_TYPE(T)
#elif defined ANYDSL2_JUST_U_TYPE
#   define ANYDSL2_U_TYPE(T) ANYDSL2_JUST_U_TYPE(T)
#   define ANYDSL2_F_TYPE(foo)
#elif defined ANYDSL2_JUST_F_TYPE
#   define ANYDSL2_U_TYPE(foo)
#   define ANYDSL2_F_TYPE(T) ANYDSL2_JUST_F_TYPE(T)
#else
#   ifndef ANYDSL2_U_TYPE
#       error "define ANYDSL2_U_TYPE before including this file"
#   endif
#   ifndef ANYDSL2_F_TYPE
#       error "define ANYDSL2_F_TYPE before including this file"
#   endif
#endif

ANYDSL2_U_TYPE(u1)
ANYDSL2_U_TYPE(u8)
ANYDSL2_U_TYPE(u16)
ANYDSL2_U_TYPE(u32)
ANYDSL2_U_TYPE(u64)

ANYDSL2_F_TYPE(f32)
ANYDSL2_F_TYPE(f64)

#undef ANYDSL2_U_TYPE
#undef ANYDSL2_F_TYPE
#undef ANYDSL2_UF_TYPE
#undef ANYDSL2_JUST_U_TYPE
#undef ANYDSL2_JUST_F_TYPE
