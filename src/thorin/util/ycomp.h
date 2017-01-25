#ifndef THORIN_BE_GRAPHS_H
#define THORIN_BE_GRAPHS_H

#include <iostream>
#include <fstream>

#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/iterator.h"
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

template<class I, class SuccFct>
class YCompScope {
public:
    YCompScope(std::ostream& ostream, YCompOrientation orientation)
        : ostream_(ostream)
    {
        ostream << "graph: {" << up_endl;
        ostream << "layoutalgorithm: compilergraph" << endl;
        ostream << "orientation: " << YCompOrientation_Names[orientation];
    }

    YCompScope(std::ostream& ostream, const Scope& scope, Range<I> range,
               SuccFct succs, YCompOrientation orientation)
        : YCompScope(ostream, orientation)
    {
        addScope(scope, range, succs);
    }

    ~YCompScope() {
        ostream() << down_endl << "}" << endl;
    }

    std::ostream& ostream() { return ostream_; }

private:
    void addScope(const Scope& scope, Range<I> range, SuccFct succs) {
        auto id = scope.id();

        auto print_node = [&] (decltype(*range.begin()) node) {
            streamf(ostream(), "node: { title: \"{}_{}\" label: \"{}\" }", node, id, node) << endl;

            for (const auto& succ : succs(node))
                streamf(ostream(), "edge: { sourcename: \"{}_{}\" targetname: \"{}_{}\" class: {} }",
                        node, id, &*succ, id, 16) << endl;
        };

        auto title = scope.entry()->unique_name();
        ostream() << "graph: {" << up_endl;
        ostream() << "title: \"" << title << "\"" << endl;
        ostream() << "label: \"" << title << "\"" << endl;

        for (auto n : range)
            print_node(n);

        ostream() << down_endl << "}";
    }

    std::ostream& ostream_;
};

template<class I, class S>
YCompScope<I, S> ycomp(std::ostream& out, YCompOrientation o, const Scope& scope, Range<I> range, S succs) {
    return YCompScope<I, S>(out, scope, range, succs, o);
}

template<class G>
void ycomp(std::ostream& out, World& world) {
    // TODO use up and down
    out << "graph: {" <<  endl;
    out << "    " << "graph: {" <<  endl;
    out << "        " << "title: \"" << world.name() << '"' << endl;
    out << "        " << "label: \"" << world.name() << '"' << endl;
    Scope::for_each(world, [&] (const Scope& scope) { G::create(scope).stream_ycomp(out); });
    out << "    " << '}' << endl;
    out << '}' << endl;
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
    void ycomp() const;                                     ///< Dumps ycomp to a file with an auto-generated file name.
    void write_ycomp(const char* filename) const;           ///< Dumps ycomp to file with name @p filename.
    virtual void stream_ycomp(std::ostream& out) const = 0; ///< Streams ycomp file to @p out.

private:
    const Scope& scope_;
    const char* name_;
};

//------------------------------------------------------------------------------

}


#endif
