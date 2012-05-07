#ifndef DSLU_ATTRIBUTES_HEADER
#define DSLU_ATTRIBUTES_HEADER

#ifdef _MSC_VER
#define ANYDSL_NORETURN __declspec(noreturn)
#else
#ifdef __GNUC__
#define ANYDSL_NORETURN __attribute__ ((noreturn))
#else
#define ANYDSL_NORETURN
#endif
#endif



#endif