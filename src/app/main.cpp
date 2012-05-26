#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <vector>

#include <boost/program_options.hpp>

#include "anydsl/literal.h"
#include "anydsl/world.h"
#include "impala/parser.h"

//------------------------------------------------------------------------------

using namespace anydsl;
using namespace std;
namespace po = boost::program_options;

typedef vector<string> Names;

//------------------------------------------------------------------------------

enum EmitType {
    None,
    AIR,
    LLVM,
};

int main(int argc, char** argv) {
    try {
        if (argc < 1)
            throw logic_error("bad number of arguments");

        string prgname = argv[0];
        Names infiles;
        string outfile = "-";
        string emittype;
        EmitType destinationType = None;
        bool help, fancy, run, notc, debug, tracing = false;

        // specify options
        po::options_description desc("Usage: " + prgname + " [options] file...");
        desc.add_options()
        ("help,h",      po::bool_switch(&help), "produce this help message")
        ("emit,e",      po::value<string>(&emittype),   "emit code, arg={air|llvm}")
        ("fancy,f",     po::bool_switch(&fancy), "use fancy air output")
        ("run,r",       po::bool_switch(&run),  "run program")
        ("notc",        po::bool_switch(&notc),  "no typechecks during execution")
        ("debug,d",     po::bool_switch(&debug), "print debug information during execution")
        ("trace,t",     po::bool_switch(&tracing), "print tracing information during execution")
        ("outfile,o",   po::value(&outfile)->default_value("-"), "specifies output file")
        ("infile,i",    po::value(&infiles),    "input file");

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

        if (!emittype.empty()) {
            if (emittype == "air")
                destinationType = AIR;
            else if (emittype == "llvm")
                ANYDSL_NOT_IMPLEMENTED;
            else
                throw logic_error("invalid emit type: " + emittype);
        }

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


        if (debug) { 
            ANYDSL_NOT_IMPLEMENTED;
        }

        World world;
        const char* filename = infiles[0].c_str();
        ifstream file(filename);
        //impala::Parser parser(world, file, filename);
        //Lambda* root = 
            //parser.parse();
        
        const Sigma* s = world.sigma2(world.type_u16(), world.type_u8());
        world.sigma3(s, s, s);
        const Sigma* u = world.sigma2(world.type_u16(), world.type_u8());
        world.sigma3(u, s, s);

        world.literal_u8(1);
        world.literal_u8(2);
        world.literal_u8(3);
        world.literal_u8(2);
        world.literal_u16(2);
        world.literal_u16(2);
        world.literal_u16(5);
        world.literal_u32(5);
        world.literal_u32(0);
        world.literal_u32(11);
        world.createArithOp(ArithOp_add, world.literal_u32(5), world.literal_u32(6));
        Value* a = world.createArithOp(ArithOp_add, world.literal_u32(11), world.literal_u32(0));
        //Value* b = world.createArithOp(ArithOp_add, world.literal_u32(7), world.literal_u32(4));
        world.createArithOp(ArithOp_add, a, world.literal_u32(6));
        world.createArithOp(ArithOp_add, world.literal_u32(6), a);
        world.createRelOp(RelOp_cmp_ult, world.literal_u16(2), world.literal_u16(5));
        world.createRelOp(RelOp_cmp_ugt, world.literal_u16(5), world.literal_u16(2));

        //FOREACH(

        //Emit the results
        switch (destinationType) {
            case None:
                break;
            case AIR:
                ANYDSL_NOT_IMPLEMENTED;
                break;
            case LLVM:
                ANYDSL_NOT_IMPLEMENTED;
                break;
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
