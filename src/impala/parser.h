#ifndef IMPALA_PARSER_H
#define IMPALA_PARSER_H

#include <istream>
#include <fstream>

namespace impala {

class Prg;
class TypeTable;

const Prg* parse(TypeTable& types, std::istream& i, const std::string& filename, bool& result);

} // namespace impala

#endif
