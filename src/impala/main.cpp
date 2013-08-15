#include <fstream>
#include <vector>
#include <cctype>

#include <boost/program_options.hpp>

#include "anydsl2/analyses/looptree.h"
#include "anydsl2/analyses/scope.h"
#include "anydsl2/analyses/verify.h"
#include "anydsl2/transform/vectorize.h"
#include "anydsl2/transform/partial_evaluation.h"
#include "anydsl2/be/air.h"
#include "anydsl2/be/il.h"
#include "anydsl2/be/llvm.h"

#include "impala/ast.h"
#include "impala/parser.h"
#include "impala/sema.h"
#include "impala/dump.h"
#include "impala/emit.h"
#include "impala/init.h"

//------------------------------------------------------------------------------

using namespace anydsl2;
using namespace std;
namespace po = boost::program_options;

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
        string outfile = "-";
        string emittype;
        bool help, emit_all, emit_air, emit_il, emit_ast, emit_llvm, emit_looptree, fancy, opt, verify, nocleanup, nossa, pe = false;
        int vectorlength = 0;

        // specify options
        po::options_description desc("Usage: " + prgname + " [options] file...");
        desc.add_options()
        ("help,h",          po::bool_switch(&help),                     "produce this help message")
        ("outfile,o",       po::value(&outfile)->default_value("-"),    "specifies output file")
        ("infile,i",        po::value(&infiles),                        "input file")
#ifndef NDEBUG
        ("break,b",         po::value(&breakpoints),                    "breakpoint at definition generation of number arg")
#endif
        ("emit-air",        po::bool_switch(&emit_air),                 "emit textual AIR representation of impala program")
        ("emit-il",         po::bool_switch(&emit_il),                  "emit textual IL representation of impala program")
        ("emit-all",        po::bool_switch(&emit_all),                 "emit AST, AIR, LLVM and loop tree")
        ("emit-ast",        po::bool_switch(&emit_ast),                 "emit AST of impala program")
        ("emit-looptree",   po::bool_switch(&emit_looptree),            "emit loop tree")
        ("emit-llvm",       po::bool_switch(&emit_llvm),                "emit llvm from AIR representation (implies -O)")
        ("fancy,f",         po::bool_switch(&fancy),                    "use fancy output")
        ("nocleanup",       po::bool_switch(&nocleanup),                "no clean-up phase")
        ("nossa",           po::bool_switch(&nossa),                    "use slots + load/store instead of SSA construction")
        ("pe",              po::bool_switch(&pe),                       "perform partial evaluation (experimantal!!!)")
        ("verify,v",        po::bool_switch(&verify),                   "run verifier")
        ("vectorize",       po::value(&vectorlength),                   "run vectorizer on main with given vector length (experimantal!!!), arg=<vector length>")
        (",O",              po::bool_switch(&opt),                      "optimize");

        // positional options, i.e., input files
        po::positional_options_description pos_desc;
        pos_desc.add("infile", -1);

        // do cmdline parsing
        po::command_line_parser clp(argc, argv);
        clp.options(desc);
        clp.positional(pos_desc);
        po::variables_map vm;
        po::store(clp.run(), vm);
        po::notify(vm);

        if (emit_all)
            emit_air = emit_looptree = emit_ast = emit_llvm = true;
        opt |= emit_llvm;

        if (infiles.empty() && !help) {
#if BOOST_VERSION >= 105000
            throw po::invalid_syntax(po::invalid_syntax::missing_parameter, "infile");
#else
            throw po::invalid_syntax("infile", po::invalid_syntax::missing_parameter);
#endif
        }

        if (help) {
            desc.print(cout);
            return EXIT_SUCCESS;
        }

        //ofstream ofs;
        //if (outfile != "-") {
            //ofs.open(outfile.c_str());
            //ofs.exceptions(istream::badbit);
        //}
        //ostream& out = ofs.is_open() ? ofs : cout;

        const char* filename = infiles[0].c_str();
        ifstream file(filename);

        impala::Init init;
#ifndef NDEBUG
        for (auto b : breakpoints) {
            assert(b.size() > 0);
            size_t num = 0;
            for (size_t i = 0, e = b.size(); i != e; ++i) {
                char c = b[i];
                if (!std::isdigit(c))
                    throw po::error("invalid breakpoint '" + b + "'");
                num = num*10 + c - '0';
            }

            init.world.breakpoint(num);
        }
#endif

        bool result;
        anydsl2::AutoPtr<const impala::Prg> p(impala::parse(init.typetable, file, filename, result));

        if (emit_ast)
            dump(p, fancy);

        result &= check(init.typetable, p, nossa);

        if (result) {
            emit(init.world, p);

            if (!nocleanup)
                init.world.cleanup();
            if (verify)
                anydsl2::verify(init.world);
            if (opt)
                init.world.opt();
            if (pe)
                partial_evaluation(init.world);
            if (vectorlength != 0) {
                Lambda* impala_main = top_level_lambdas(init.world)[0];
                Scope scope(impala_main);
                anydsl2::vectorize(scope, vectorlength);
            }
            if (emit_air)
                anydsl2::emit_air(init.world, fancy);
            if (emit_il)
                anydsl2::emit_il(init.world, fancy);
            if (emit_looptree)
                std::cout << Scope(init.world).looptree().root() << std::endl; // TODO

            if (emit_llvm)
                anydsl2::emit_llvm(init.world);
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
