#include <fstream>
#include <vector>
#include <cctype>
#include <stdexcept>

#include "thorin/analyses/looptree.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/verify.h"
#include "thorin/transform/import.h"
#include "thorin/transform/vectorize.h"
#include "thorin/transform/partial_evaluation.h"
#include "thorin/be/thorin.h"
#include "thorin/be/il.h"
#include "thorin/be/llvm/llvm.h"
#include "thorin/util/args.h"

#include "impala/ast.h"
#include "impala/parser.h"
#include "impala/sema.h"
#include "impala/dump.h"
#include "impala/emit.h"
#include "impala/init.h"

//------------------------------------------------------------------------------

using namespace thorin;
using namespace std;

typedef vector<string> Names;

//------------------------------------------------------------------------------

int main(int argc, char** argv) {
    try {
        if (argc < 1)
            throw logic_error("bad number of arguments");

        string prgname = argv[0];
        Names infiles;
#ifndef NDEBUG
        Names breakpoints;
#endif
        string outfile;
        bool help, emit_all, emit_thorin, emit_il, emit_ast, emit_llvm, emit_looptree, fancy, nocolor, opt, verify, nocleanup, nossa = false;
        int vectorlength = 0;
        auto cmd_parser = ArgParser()
            .implicit_option("infiles", "input files", infiles)
            // specify options
            .add_option<bool>("help", "produce this help message", help, false)
            .add_option<string>("o", "specifies the output file", outfile, "-")
#ifndef NDEBUG
            .add_option<vector<string>>("break", "breakpoint at definition generation of number arg", breakpoints)
#endif
            .add_option<bool>("nocleanup", "no clean-up phase", nocleanup, false)
            .add_option<bool>("nossa", "use slots + load/store instead of SSA construction", nossa, false)
            .add_option<bool>("verify", "run verifier", verify, false)
            .add_option<int>("vectorize", "run vectorizer on main with given vector length (experimantal!!!), arg=<vector length>", vectorlength, false)
            .add_option<bool>("emit-air", "emit textual THORIN representation of impala program", emit_thorin, false) // legacy support
            .add_option<bool>("emit-thorin", "emit textual THORIN representation of impala program", emit_thorin, false)
            .add_option<bool>("emit-il", "emit textual IL representation of impala program", emit_il, false)
            .add_option<bool>("emit-all", "emit AST, AIR, LLVM and loop tree", emit_all, false)
            .add_option<bool>("emit-ast", "emit AST of impala program", emit_ast, false)
            .add_option<bool>("emit-looptree", "emit loop tree", emit_looptree, false)
            .add_option<bool>("emit-llvm", "emit llvm from AIR representation (implies -O)", emit_llvm, false)
            .add_option<bool>("f", "use fancy output", fancy, false)
            .add_option<bool>("nc", "use uncolored output", nocolor, false)
            .add_option<bool>("O", "optimize", opt, false);

        // do cmdline parsing
        cmd_parser.parse(argc, argv);

        if (emit_all)
            emit_thorin = emit_looptree = emit_ast = emit_llvm = true;
        opt |= emit_llvm;

        if (infiles.empty() && !help) {
            std::cerr << "no input files" << std::endl;
            return EXIT_FAILURE;
        }

        if (help) {
            std::cout << "Usage: " + prgname + " [options] file..." << std::endl;
            cmd_parser.print_help();
            return EXIT_SUCCESS;
        }

        std::string module_name;
        for (auto infile : infiles) {
            auto i = infile.find_last_of('.');
            if (infile.substr(i + 1) != "impala")
                throw logic_error("input file '" + infile + "' does not have '.impala' extension");
            auto rest = infile.substr(0, i);
            if (rest.empty())
                throw logic_error("input file '" + infile + "' has empty module name");
            module_name = rest;
        }

        impala::Init init(module_name);

#ifndef NDEBUG
        for (auto b : breakpoints) {
            assert(b.size() > 0);
            size_t num = 0;
            for (size_t i = 0, e = b.size(); i != e; ++i) {
                char c = b[i];
                if (!std::isdigit(c)) {
                    std::cerr << "invalid breakpoint '" << b << "'" << std::endl;
                    return EXIT_FAILURE;
                }
                num = num*10 + c - '0';
            }

            init.world.breakpoint(num);
        }
#endif

        thorin::AutoPtr<impala::Scope> prg = new impala::Scope();
        prg->set_loc(thorin::Location(infiles[0], 1, 1, 1, 1));

        bool result = true;
        for (auto infile : infiles) {
            std::string filename = infile.c_str();
            ifstream file(filename);
            result &= impala::parse(init.typetable, file, filename, prg);
        }

        if (emit_ast)
            dump_prg(prg, fancy);

        result &= check(init.typetable, prg, nossa);
        result &= result ? emit(init.world, prg) : false;

        if (result) {
            if (!nocleanup)
                init.world.cleanup();
            if (verify)
                thorin::verify(init.world);
            if (opt)
                init.world.opt();
            if (vectorlength != 0) {
                Lambda* impala_main = top_level_lambdas(init.world)[0];
                Scope scope(impala_main);
                thorin::vectorize(scope, vectorlength);
                init.world.cleanup();
            }
            if (emit_thorin)
                thorin::emit_thorin(init.world, fancy, !nocolor);
            if (emit_il)
                thorin::emit_il(init.world, fancy);
            if (emit_looptree) {
                for (auto top : top_level_lambdas(init.world)) {
                    Scope scope(top);
                    const LoopTree looptree(scope);
                    std::cout << looptree.root() << std::endl; // TODO
                }
            }

            if (emit_llvm)
                thorin::emit_llvm(init.world);
        }

        return EXIT_SUCCESS;
    } catch (exception const& e) {
        cerr << e.what() << endl;
        return EXIT_FAILURE;
    } catch (...) {
        cerr << "unknown exception" << endl;
        return EXIT_FAILURE;
    }
}
