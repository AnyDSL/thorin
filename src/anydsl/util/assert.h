#ifndef DSLU_ASSERT_CLASS_HEADER
#define DSLU_ASSERT_CLASS_HEADER

#include <cstdlib>
#include <iostream>

#ifdef _MSC_VER
#define ANYDSL_NORETURN __declspec(noreturn)
#else
#ifdef __GNUC__
#define ANYDSL_NORETURN __attribute__ ((noreturn))
#else
#define ANYDSL_NORETURN
#endif
#endif



namespace anydsl {

enum BTEnum {
    AssertBacktrace
};

namespace detail {

#ifdef __GLIBC__
void glibc_printBacktrace();
#endif
#ifdef _WIN32
void win32_printDebugString(const char* errorMsg);
#endif

class Assert {
public:
#ifndef NDEBUG
    Assert(bool cond, bool warn=false)
        : result(cond),
          warning(warn)
    {}

    Assert(bool cond, const char* message)
        : result(cond) {
        if (!result)
            std::cerr << message << std::endl;
    }

    ~Assert() {
        if (!result) {
            std::cerr << std::endl;
            if (!warning) {
#ifndef _WIN32
                glibc_printBacktrace();
#endif
                abort();
            }
        }
    }
#endif


    template <class T>
    inline Assert& operator<<(const T& message) {
#ifndef NDEBUG
        if (!result)
            std::cerr << message;
#endif
        return *this;
    }


private:
#ifndef NDEBUG
    bool result;
    bool warning;
#endif
};

class AlwaysAssert {
public:
#ifndef NDEBUG
    AlwaysAssert() {}

    AlwaysAssert(const char* message) {
        std::cerr << message << std::endl;
    }

    ANYDSL_NORETURN inline ~AlwaysAssert() {
        std::cerr << std::endl;
#ifndef _WIN32
        glibc_printBacktrace();
#endif
        abort();
    }
#endif

    template <class T>
    inline AlwaysAssert& operator<<(const T& message) {
#ifndef NDEBUG
        std::cerr << message;
#endif
        return *this;
    }


private:
};

#ifndef NDEBUG
template <>
inline Assert& Assert::operator<< <BTEnum>(const BTEnum& message) {
    if (!result) {
        std::cerr << message << std::endl;
#ifndef _WIN32
        glibc_printBacktrace();
#endif
    }
    return *this;
}
#endif

}

#ifdef assert
#undef assert
#endif

#define DSLU_STRING(s) #s
#define DSLU_CONCAT_BOO(x,y) x ## y
#define DSLU_CONCAT(x,y) DSLU_CONCAT_BOO(x,y)
#define DSLU_UNIQUE(name) DSLU_CONCAT(name , __LINE__)

#ifndef NDEBUG
#define assert( cond ) \
    ::anydsl::detail::Assert( cond ) << "Assertion failure at " << __FILE__ << ":" << __LINE__ << " -- " DSLU_STRING(cond)

#define anydsl_assert( cond , msg ) \
    ::anydsl::detail::Assert( cond ) << "Assertion failure at " << __FILE__ << ":" << __LINE__ << " -- " << msg

#define anydsl_assert_always( msg ) \
    ::anydsl::detail::AlwaysAssert() << "Assertion failure at " << __FILE__ << ":" << __LINE__ << " -- " << msg


#define warn_assert( cond ) \
{ \
    static bool DSLU_UNIQUE(warn_assert_once) = true; \
    bool DSLU_UNIQUE(warn_assert_oncelaunch) = DSLU_UNIQUE(warn_assert_once); \
    bool DSLU_UNIQUE(warn_assert_cond) = (cond); \
    if (!DSLU_UNIQUE(warn_assert_cond)) \
        DSLU_UNIQUE(warn_assert_once) = false; \
    if (DSLU_UNIQUE(warn_assert_oncelaunch)) \
        ::anydsl::detail::Assert( DSLU_UNIQUE(warn_assert_cond) , true ) << "Warning assertion at " << __FILE__ << ":" << __LINE__ << " -- " DSLU_STRING(cond); \
}

#define anydsl_warn_assert( cond , msg )\
{ \
    static bool DSLU_UNIQUE(warn_assert_once) = true; \
    bool DSLU_UNIQUE(warn_assert_oncelaunch) = DSLU_UNIQUE(warn_assert_once); \
    bool DSLU_UNIQUE(warn_assert_cond) = (cond); \
    if ( !DSLU_UNIQUE(warn_assert_cond) ) \
        DSLU_UNIQUE(warn_assert_once) = false; \
    if ( DSLU_UNIQUE(warn_assert_oncelaunch) ) \
        ::anydsl::detail::Assert( DSLU_UNIQUE(warn_assert_cond) , true ) << "Warning assertion at " << __FILE__ << ":" << __LINE__ << " -- " << msg; \
}

#else
#define assert( cond ) \
    ::anydsl::detail::Assert()
#define anydsl_assert( cond , msg ) \
    ::anydsl::detail::Assert()
#define warn_assert( cond ) \
    ::anydsl::detail::Assert()
#define anydsl_warn_assert( cond , msg ) \
    ::anydsl::detail::Assert()
#define anydsl_assert_always( msg ) \
    ::anydsl::detail::Assert()
#endif

#ifndef NDEBUG
#define ANYDSL_CALL_ONCE
#else
#define ANYDSL_CALL_ONCE do { static bool once = true; anydsl_assert(once,"ANYDSL_CALL_ONCE"); once=false; } while(0)
#endif
#define ANYDSL_NOT_IMPLEMENTED do { anydsl_assert_always("Function not implemented"); abort(); } while(0)
#define ANYDSL_DEPRECATED do { anydsl_warn_assert(false, "Function is deprecated and should not be used"); } while(0)
#define ANYDSL_DEPRECATED_BT do { anydsl_warn_assert(false, "Function is deprecated and should not be used") << ::anydsl::AssertBacktrace; } while(0)

#ifndef _MSC_VER
#define ANYDSL_UNREACHABLE do { anydsl_assert_always("unreachable"); abort(); } while(0)
#else
inline __declspec(noreturn) void anydsl_dummy_function() { abort(); }
#define ANYDSL_UNREACHABLE do { anydsl_assert_always("unreachable"); anydsl::anydsl_dummy_function(); } while(0)
#endif

}

#endif
