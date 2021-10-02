#include <cstring>
#include <iostream>
#include <fstream>

#include "thorin/fe/parser.h"

using namespace thorin;

static const auto usage =
"Usage: thorin [options] file\n"
"\n"
"Options:\n"
"\t-h, --help\tdisplay this help and exit\n"
"\t-v, --version\tdisplay version info and exit\n"
"\n"
"Hint: use '-' as file to read from stdin.\n"
;

static const auto version = "thorin command-line utility 0.1\n";

int main(int argc, char** argv) {
    try {
        const char* file = nullptr;

        for (int i = 1; i != argc; ++i) {
            if (strcmp("-h", argv[i]) == 0 || strcmp("--help", argv[i]) == 0) {
                std::cerr << usage;
                return EXIT_SUCCESS;
            } else if (strcmp("-v", argv[i]) == 0 || strcmp("--version", argv[i]) == 0) {
                std::cerr << version;
                return EXIT_SUCCESS;
            } else if (file == nullptr) {
                file = argv[i];
            } else {
                throw std::logic_error("multiple input files given");
            }
        }

        if (file == nullptr)
            throw std::logic_error("no input file given");

        World world;
        if (strcmp("-", file) == 0) {
            Parser parser(world, "<stdin>", std::cin);
            //exp = parser.parse_prg();
        } else {
            std::ifstream ifs(file);
            Parser parser(world, file, ifs);
            //exp = parser.parse_prg();
        }

        //if (num_errors != 0) {
            //std::cerr << num_errors << " error(s) encountered" << std::endl;
            //return EXIT_FAILURE;
        //}

        //if (eval) exp = exp->eval();
        //exp->dump();
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        std::cerr << usage;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "error: unknown exception" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
