#ifndef IMPALA_PARSER_H
#define IMPALA_PARSER_H

#include <istream>
#include <fstream>


namespace impala {

class Prg;

const Prg* parse(std::istream& i, const std::string& filename);

} // namespace impala

#endif
