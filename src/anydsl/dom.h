#ifndef ANYDSL_DOM_H
#define ANYDSL_DOM_H

#include <boost/unordered_map.hpp>

#include "anydsl/util/array.h"

namespace anydsl {

class Def;
class PostOrder;

class Dominators {
public:
    Dominators(const Def* def);
    Dominators(const Def* def, const PostOrder& order);
    ~Dominators();

    typedef boost::unordered_map<const Def*, const Def*> DominatorRelation;
    typedef boost::unordered_multimap<const Def*, const Def*> DomChildren;
    typedef DominatorRelation::const_iterator const_iterator;
    typedef std::pair<DomChildren::const_iterator, DomChildren::const_iterator> Range;

    const_iterator begin() const { return relation.begin(); }
    const_iterator end() const { return relation.end(); }

    const DomChildren& children() const { return children_; }

    const Def* operator[](const Def* source) const {
        DominatorRelation::const_iterator it = relation.find(source);
        if(it != relation.end())
            return it->second;
        return 0;
    }

private:
    typedef Array<int> Doms;

    void init(const Def* def, const PostOrder& order);
    int intersect(int first, int second, const Doms& doms);

    DominatorRelation relation;
    DomChildren children_;
};

}

#endif
