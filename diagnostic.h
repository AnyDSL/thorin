// #define _GNU_SOURCE /* See feature_test_macros(7) */
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <ostream>
#include <stdio.h>
#include <string.h>
#include <iostream>

#include "printable.h"

/*static void fpututf32(utf32 const c, FILE *const out) {
if (c < 0x80U) {
fputc(c, out);
} else if (c < 0x800) {
fputc(0xC0 | (c >> 6), out);
fputc(0x80 | (c & 0x3F), out);
} else if (c < 0x10000) {
fputc(0xE0 | ( c >> 12), out);
fputc(0x80 | ((c >>  6) & 0x3F), out);
fputc(0x80 | ( c        & 0x3F), out);
} else {
fputc(0xF0 | ( c >> 18), out);
fputc(0x80 | ((c >> 12) & 0x3F), out);
fputc(0x80 | ((c >>  6) & 0x3F), out);
fputc(0x80 | ( c        & 0x3F), out);
}
}*/

static char* make_message(const char *fmt, va_list ap) {
   int n;
   int size = 100;     /* Guess we need no more than 100 bytes */
   char *p, *np;

   if ((p = malloc(size)) == NULL)
       return NULL;

   while (1) {
       /* Try to print in the allocated space */
       //va_start(ap, fmt);
       n = vsnprintf(p, size, fmt, ap);
       //va_end(ap);

       /* Check error code */
       if (n < 0)
           return NULL;

       /* If that worked, return the string */
       if (n < size)
           return p;

       /* Else try again with more space */
       size = n + 1;       /* Precisely what is needed */

       if ((np = realloc (p, size)) == NULL) {
           free(p);
           return NULL;
       } else {
           p = np;
       }
   }
}

static inline char const* strstart(char const* str, char const* start) {
	do {
		if (*start == '\0')
			return str;
	} while (*str++ == *start++);
	return NULL;
}

static void print(std::ostream &out, char const *fmt, ...) {
    char *tmp;

    va_list argp;
    va_start(argp, fmt);
#ifdef _GNU_SOURCE
    vasprintf(&tmp, fmt, argp);
#else
    tmp = make_message(fmt, argp); 
#endif
    out << tmp;
    va_end(argp);

    free(tmp);
}

/**
 * Issue a diagnostic message.
 */
static void messagevf(std::ostream &out, char const *fmt, va_list ap) {
    //FILE *const out = stderr;

    for (char const *f; (f = strchr(fmt, '%')); fmt = f) {
        for(unsigned int i = 0; i < f - fmt; i++) {
            out << fmt[i];
        }

        //fwrite(fmt, sizeof(*fmt), f - fmt, out); // Print till '%'.
        ++f; // Skip '%'.

        bool extended = false;
        bool flag_zero = false;
        bool flag_long = false;
        bool flag_high = false;
        for (; ; ++f) {
            switch (*f) {
                case '#':
                    extended = true;
                    break;
                case '0':
                    flag_zero = true;
                    break;
                case 'l':
                    flag_long = true;
                    break;
                case 'h':
                    flag_high = true;
                    break;
                default:
                    goto done_flags;
            }
        }
        done_flags:;

        int field_width = 0;
        if (*f == '*') {
            ++f;
            field_width = va_arg(ap, int);
        }

        int precision = -1;
        char const *const rest = strstart(f, ".*");
        if (rest) {
            f = rest;
            precision = va_arg(ap, int);
        }

        /* Automatic highlight for some formats. */
        /*if (!flag_high)
            flag_high = strchr("EKNQTYk", *f);

        if (flag_high)
            fputs(colors.highlight, out);*/
        switch (*f++) {
            case '%':
                out << '%';
                //fputc('%', out);
                break;

            case 'c': {
                /*if (flag_long) {
                    const utf32 val = va_arg(ap, utf32);
                    fpututf32(val, out);
                } else {*/
                    const unsigned char val = (unsigned char) va_arg(ap, int);
                    out << val;
                    //fputc(val, out);
                //}
                break;
            }
            
            case 'i':
            case 'd': {
                if (flag_long) {
                    const long val = va_arg(ap, long);
                    print(out, "%ld", val);
                    //fprintf(out, "%ld", val);
                } else {
                    const int val = va_arg(ap, int);
                    print(out, "%d", val);
                    //fprintf(out, "%d", val);
                }
                break;
            }

            case 's': {
                const char *const str = va_arg(ap, const char*);
                print(out, "%.*s", precision, str);
                //fprintf(out, "%.*s", precision, str);
                break;
            }

            case 'S': {
                const std::string *str = va_arg(ap, const std::string*);
		//const string_t *str = va_arg(ap, const string_t*);
                out << str;
                //fwrite(str->begin, 1, str->size, out);
                break;
            }

            case 'u': {
                const unsigned int val = va_arg(ap, unsigned
                        int);
                print(out, "%u", val);
                //fprintf(out, "%u", val);
                break;
            }

            case 'X': {
                unsigned int const val = va_arg(ap, unsigned
                        int);
                char const *const xfmt = flag_zero ? "%0*X" : "%*X";
                print(out, xfmt, field_width, val);
                //fprintf(out, xfmt, field_width, val);
                break;
            }

            case 'Y': {
                const Printable *p = va_arg(ap, const Printable*);
                p->print(out);
                break;
            }

            /*case 'Y': {
                const symbol_t *const symbol = va_arg(ap, const symbol_t*);
                out << symbol ? symbol->string : "(null)";
                //fputs(symbol ? symbol->string : "(null)", out);
                break;
            }

            case 'E': {
                const expression_t *const expr = va_arg(ap, const expression_t*);
                print_expression(expr);
                break;
            }

            case 'Q': {
                const unsigned qualifiers = va_arg(ap, unsigned);
                print_type_qualifiers(qualifiers, QUAL_SEP_NONE);
                break;
            }

            case 'T': {
                const type_t *const type = va_arg(ap, const type_t*);
                print_type_ext(type, NULL, NULL);
                break;
            }

            case 'K': {
                const token_t *const token = va_arg(ap, const token_t*);
                print_token(out, token);
                break;
            }

            case 'k': {
                if (flag_long) {
                    va_list *const toks = va_arg(ap, va_list *);
                    separator_t sep = {"", ", "};
                    for (; ;) {
                        const token_kind_t tok = (token_kind_t) va_arg(*toks, int);
                        if (tok == 0)
                            break;
                        fputs(sep_next(&sep), out);
                        fputc('\'', out);
                        print_token_kind(out, tok);
                        fputc('\'', out);
                    }
                } else {
                    const token_kind_t token = (token_kind_t) va_arg(ap, int);
                    fputc('\'', out);
                    print_token_kind(out, token);
                    fputc('\'', out);
                }
                break;
            }

            case 'N': {
                entity_t const *
                const ent = va_arg(ap, entity_t const*);
                if (extended && is_declaration(ent)) {
                    print_type_ext(ent->declaration.type, ent->base.symbol, NULL);
                } else {
                    char const *const ent_kind = get_entity_kind_name(ent->kind);
                    symbol_t const *const sym = ent->base.symbol;
                    if (sym) {
                        fprintf(out, "%s %s", ent_kind, sym->string);
                    } else {
                        fprintf(out, "anonymous %s", ent_kind);
                    }
                }
                break;
            }*/

            default:
                std::cerr << "unknown format specifier";
		std::cerr << *(f - 1);
        }
        /*if (flag_high)
            fputs(colors.reset_highlight, out);*/
    }
    out << fmt;
    //fputs(fmt, out); // Print rest.
}
