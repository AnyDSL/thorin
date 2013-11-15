#ifdef THORIN_UF_TYPE
#   define THORIN_U_TYPE(T) THORIN_UF_TYPE(T)
#   define THORIN_F_TYPE(T) THORIN_UF_TYPE(T)
#elif defined THORIN_JUST_U_TYPE
#   define THORIN_U_TYPE(T) THORIN_JUST_U_TYPE(T)
#   define THORIN_F_TYPE(foo)
#elif defined THORIN_JUST_F_TYPE
#   define THORIN_U_TYPE(foo)
#   define THORIN_F_TYPE(T) THORIN_JUST_F_TYPE(T)
#else
#   ifndef THORIN_U_TYPE
#       error "define THORIN_U_TYPE before including this file"
#   endif
#   ifndef THORIN_F_TYPE
#       error "define THORIN_F_TYPE before including this file"
#   endif
#endif

THORIN_U_TYPE(u1)
THORIN_U_TYPE(u8)
THORIN_U_TYPE(u16)
THORIN_U_TYPE(u32)
THORIN_U_TYPE(u64)

THORIN_F_TYPE(f32)
THORIN_F_TYPE(f64)

#undef THORIN_U_TYPE
#undef THORIN_F_TYPE
#undef THORIN_UF_TYPE
#undef THORIN_JUST_U_TYPE
#undef THORIN_JUST_F_TYPE
