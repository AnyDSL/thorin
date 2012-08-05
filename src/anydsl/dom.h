#ifndef ANYDSL_DOM_H
#define ANYDSL_DOM_H

#include <boost/unordered_map.hpp>

#include "anydsl/def.h"
#include "anydsl/util/for_all.h"
#include "anydsl/order.h"
#include "anydsl/util/array.h"

namespace anydsl {

class Dominators {
public:
    Dominators(const Def* def);
    Dominators(const Def* def, const PostOrder& order);
    ~Dominators();

    typedef boost::unordered_map<const Def*, const Def*> DominatorRelation;
    typedef DominatorRelation::const_iterator iterator;

    iterator begin() const { return relation.begin(); }
    iterator end() const { return relation.end(); }

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
};

}

#endif
