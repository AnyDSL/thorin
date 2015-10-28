#ifndef THORIN_BE_GRAPHS_H
#define THORIN_BE_GRAPHS_H

#include <iostream>
#include <fstream>

#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/iterator.h"
#include "thorin/util/printer.h"
#include "thorin/util/stream.h"

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
    LeftToRight = 0,
    RightToLeft,
    BottomToTop,
    TopToBottom,
    Num
};

static const char* YComp_Orientation_Names[] = { "left_to_right", "right_to_left", "bottom_to_top", "top_to_bottom" };
static_assert(sizeof(YComp_Orientation_Names)/sizeof(char*) == YComp_Orientation::Num, "Sizes do not match!");

struct YCompConfig {
    static int indentation;
};

template <typename I, typename SuccFct>
class YCompScope : public Printer {
public:
    YCompScope(std::ostream& ostream, const Scope& scope, Range<I> range,
               SuccFct succs, YComp_Orientation orientation)
        : YCompScope(ostream, orientation)
    {
        addScope(scope, range, succs);
    }

    ~YCompScope() {
        down() << "}";
        indent -= YCompConfig::indentation;
        newline();
    }

private:
    YCompScope(std::ostream& ostream, YComp_Orientation orientation)
            : Printer(ostream)
    {
        indent += YCompConfig::indentation;

        newline() << "graph: {";
        up() << "layoutalgorithm: compilergraph";
        newline() << "orientation: " << YComp_Orientation_Names[orientation];
    }

    void addScope(const Scope& scope, Range<I> range, SuccFct succs) {
        auto id = scope.id();

        auto print_node = [&] (decltype(*range.begin()) node) {
            newline() << "node: { title: \"" << node << "_" << id << "\" label: \"" << node << "\" }";

            for (auto succ : succs(node)) {
                newline() << "edge: { sourcename: \"" << node << "_" << id
                          << "\" targetname: \"" << succ << "_" << id << "\" class: " << 16 << " }";
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

template <typename I, typename SuccFct>
YCompScope<I, SuccFct> emit_ycomp(std::ostream& ostream, const Scope& scope, Range<I> range, SuccFct succs,
                                           YComp_Orientation orientation = YComp_Orientation::BottomToTop) {
    return YCompScope<I, SuccFct>(ostream, scope, range, succs, orientation);
}

template<class G>
void emit_ycomp(std::ostream& out, World& world, void (G::* ycomp)(std::ostream&) const) {
    out << "graph: {" <<  std::endl;
    out << "    " << "graph: {" <<  std::endl;
    out << "        " << "title: \"" << world.name() << '"' << std::endl;
    out << "        " << "label: \"" << world.name() << '"' << std::endl;
    YCompConfig::indentation = 2;
    Scope::for_each(world, [&] (const Scope& scope) { G::get(scope).ycomp(out); });
    YCompConfig::indentation = 0;
    out << "    " << '}' << std::endl;
    out << '}' << std::endl;
}

}

#endif
