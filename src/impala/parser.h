#ifndef IMPALA_PARSER_H
#define IMPALA_PARSER_H

#include <istream>

namespace impala {

class Prg;
class World;

const Prg* parse(World& world, std::istream& i, const std::string& filename, bool& result);

} // namespace impala

#endif
