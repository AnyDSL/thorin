#ifndef THORIN_GRAPHS_H
#define THORIN_GRAPHS_H

#include <iostream>
#include <fstream>

#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/iterator.h"
#include "thorin/util/printer.h"

namespace thorin {

class YCompCommandLine {
public:
    YCompCommandLine() {}

    void add(std::string graph, bool temp, std::string file);
    void print(World& world);

private:
    std::vector<std::string> graphs;
    std::vector<bool> temps;
    std::vector<std::string> files;

};

enum YComp_Orientation {
    LEFT_TO_RIGHT = 0,
    RIGHT_TO_LEFT,
    BOTTOM_TO_TOP,
    TOP_TO_BOTTOM,
    SIZE_OF_ENUM
};

static const char* YComp_Orientation_Names[] = { "left_to_right", "right_to_left", "bottom_to_top", "top_to_bottom" };
static_assert(sizeof(YComp_Orientation_Names)/sizeof(char*) == YComp_Orientation::SIZE_OF_ENUM, "Sizes do not match!");

struct YCompConfig {
    static int INDENTATION;
};

template <typename I, typename SuccFct, typename UniqueFct>
class YCompScope : public Printer {
public:
    YCompScope(std::ostream& ostream, const Scope& scope, Range<I> range, 
               SuccFct succs, UniqueFct unique, YComp_Orientation orientation)
        : YCompScope(ostream, orientation)
    {
        addScope(scope, range, succs, unique);
    }

    ~YCompScope() {
        down() << "}";
        indent -= YCompConfig::INDENTATION;
        newline();
    }

private:
    YCompScope(std::ostream& ostream, YComp_Orientation orientation)
            : Printer(ostream)
    {
        indent += YCompConfig::INDENTATION;

        newline() << "graph: {";
        up() << "layoutalgorithm: compilergraph";
        newline() << "orientation: " << YComp_Orientation_Names[orientation];
    }

    void addScope(const Scope& scope, Range<I> range, SuccFct succs, UniqueFct unique) {
        auto id = scope.id();

        auto print_node = [&] (decltype(*range.begin()) node) {
            auto nodeStrings = unique(node);
            newline() << "node: { title: \"" << nodeStrings.first << "_" << id << "\" label: \"" << nodeStrings.second << "\" }";

            for(auto child : succs(node)) {
                newline() << "edge: { sourcename: \"" << nodeStrings.first << "_" << id 
                          << "\" targetname: \"" << unique(child).first << "_" << id << "\" class: " << 16 << " }";
            }
        };

        auto title = scope.entry()->unique_name();
        newline() << "graph: {";
        up() << "title: \"" << title << "\"";
        newline() << "label: \"" << title << "\"";

        for (auto n : range)
            print_node(n);

        down() << "}";
    }
};

template <typename I, typename SuccFct, typename UniqueFct>
YCompScope<I,SuccFct,UniqueFct> emit_ycomp(std::ostream& ostream, const Scope& scope, Range<I> range, 
                                           SuccFct succs, UniqueFct unique, 
                                           YComp_Orientation orientation = YComp_Orientation::BOTTOM_TO_TOP) {
    return YCompScope<I,SuccFct,UniqueFct>(ostream, scope, range, succs, unique, orientation);
}

template<class Emit>
void emit_ycomp(std::ostream& ostream, const World& world, Emit emit) {
    ostream << "graph: {" <<  std::endl;
    ostream << "    " << "graph: {" <<  std::endl;
    ostream << "        " << "title: \"" << world.name() << '"' << std::endl;
    ostream << "        " << "label: \"" << world.name() << '"' << std::endl;
    YCompConfig::INDENTATION = 2;
    Scope::for_each(world, [&] (const Scope& scope) { emit(scope, ostream); });
    YCompConfig::INDENTATION = 0;
    ostream << "    " << '}' << std::endl;
    ostream << '}' << std::endl;
}

}

#endif
