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

enum YCompOrientation {
    LeftToRight = 0,
    RightToLeft,
    BottomToTop,
    TopToBottom,
    Num
};

static const char* YCompOrientation_Names[] = { "left_to_right", "right_to_left", "bottom_to_top", "top_to_bottom" };
static_assert(sizeof(YCompOrientation_Names)/sizeof(char*) == YCompOrientation::Num, "Sizes do not match!");

struct YCompConfig {
    static int indentation;
};

template <typename I, typename SuccFct>
class YCompScope : public Printer {
public:
    YCompScope(std::ostream& ostream, const Scope& scope, Range<I> range,
               SuccFct succs, YCompOrientation orientation)
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
    YCompScope(std::ostream& ostream, YCompOrientation orientation)
            : Printer(ostream)
    {
        indent += YCompConfig::indentation;

        newline() << "graph: {";
        up() << "layoutalgorithm: compilergraph";
        newline() << "orientation: " << YCompOrientation_Names[orientation];
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

template <typename I, typename S>
YCompScope<I, S> ycomp(std::ostream& out, YCompOrientation o, const Scope& scope, Range<I> range, S succs) {
    return YCompScope<I, S>(out, scope, range, succs, o);
}

template<class G>
void ycomp(std::ostream& out, World& world) {
    out << "graph: {" <<  std::endl;
    out << "    " << "graph: {" <<  std::endl;
    out << "        " << "title: \"" << world.name() << '"' << std::endl;
    out << "        " << "label: \"" << world.name() << '"' << std::endl;
    YCompConfig::indentation = 2;
    Scope::for_each(world, [&] (const Scope& scope) { G::create(scope).stream_ycomp(out); });
    YCompConfig::indentation = 0;
    out << "    " << '}' << std::endl;
    out << '}' << std::endl;
}

//------------------------------------------------------------------------------

class YComp {
public:
    YComp(const Scope& scope, const char* name)
        : scope_(scope)
        , name_(name)
    {}

    virtual ~YComp() {}

    const Scope& scope() const { return scope_; }
    const World& world() const { return scope().world(); }
    const char* name() const { return name_; }
    // Note that we don't use overloading for the following methods in order to have them accessible from gdb.
    void ycomp() const;                                     ///< Dumps ycomp to a file with an auto-generated a file name.
    void write_ycomp(const char* filename) const;           ///< Dumps ycomp file to @p filename.
    virtual void stream_ycomp(std::ostream& out) const = 0; ///< Dumps ycomp file to @p out.

private:
    const Scope& scope_;
    const char* name_;
};

//------------------------------------------------------------------------------

}


#endif
