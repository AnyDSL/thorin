#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <vector>

#include <boost/program_options.hpp>
#include <boost/array.hpp>

#include "anydsl/literal.h"
#include "anydsl/lambda.h"
#include "anydsl/dump.h"
#include "anydsl/primop.h"
#include "anydsl/world.h"
#include "impala/parser.h"
#include "impala/ast.h"
#include "impala/sema.h"
#include "impala/dump.h"
#include "impala/emit.h"
#include "impala/type.h"
#include "impala/init.h"

//------------------------------------------------------------------------------

using namespace anydsl;
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
        string outfile = "-";
        string emittype;
        bool help, emit_air, emit_ast, emit_dot, fancy = false;

        // specify options
        po::options_description desc("Usage: " + prgname + " [options] file...");
        desc.add_options()
        ("help,h",      po::bool_switch(&help),                     "produce this help message")
        ("emit-air",    po::bool_switch(&emit_air),                 "emit textual AIR representation of impala program")
        ("emit-ast",    po::bool_switch(&emit_ast),                 "emit AST of impala program")
        ("emit-dot",    po::bool_switch(&emit_dot),                 "emit dot, arg={air|llvm}")
        ("fancy,f",     po::bool_switch(&fancy),                    "use fancy output")
        ("outfile,o",   po::value(&outfile)->default_value("-"),    "specifies output file")
        ("infile,i",    po::value(&infiles),                        "input file");

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

        if (infiles.empty() && !help)
            throw po::invalid_syntax("infile", po::invalid_syntax::missing_parameter);

        if (help) {
            desc.print(cout);
            return EXIT_SUCCESS;
        }

        ofstream ofs;
        if (outfile != "-") {
            ofs.open(outfile.c_str());
            ofs.exceptions(istream::badbit);
        }
        //ostream& out = ofs.is_open() ? ofs : cout;

        const char* filename = infiles[0].c_str();
        ifstream file(filename);
        impala::init();
        impala::TypeTable types;
        anydsl::AutoPtr<const impala::Prg> p(impala::parse(types, file, filename));
        check(types, p);

        if (emit_ast)
            dump(p, fancy);
        if (emit_dot)
            ANYDSL_NOT_IMPLEMENTED;

        World world;
        Lambda* l = new Lambda();
        Param* pa = l->appendParam(world.type_u32());
        Param* pb = l->appendParam(world.type_u32());
        pa->debug = "a";
        pb->debug = "b";
        l->calcType(world);
        const Def* args[] = { world.literal_u32(42), world.literal_u32(23) };
        const Jump* jump = world.createJump(l, args);
        l->setJump(jump);
        const Lambda* newl = world.finalize(l, true);
        dump(newl);
        std::cout << std::endl;
        const Value* add = world.createArithOp(ArithOp_add,  pa, world.literal_u32(42));
        const Value* mul = world.createArithOp(ArithOp_mul, add, world.literal_u32(23));
        mul->dump();
        std::cout << std::endl;

        if (emit_air) {
            emit(world, p);
        }

        impala::destroy();
        
        return EXIT_SUCCESS;
    } catch (exception const& e) {
        cerr << e.what() << endl;
        return EXIT_FAILURE;
    } catch (...) {
        cerr << "unknown exception" << endl;
        return EXIT_FAILURE;
    }
}
