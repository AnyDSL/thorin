#ifndef ANYDSL_ORDER_H
#define ANYDSL_ORDER_H

#include <vector>
#include <boost/unordered_map.hpp>

#include "anydsl/def.h"

namespace anydsl {

class PostOrder {
public:
    typedef std::vector<const Def*> PostOrderList;
    typedef PostOrderList::const_iterator iterator;
    typedef PostOrderList::const_reverse_iterator reverse_iterator;

    PostOrder(const Def* def);
    ~PostOrder();

    void reset();

    iterator begin() const { return list_.begin(); }
    iterator end() const { return list_.end(); }

    reverse_iterator rbegin() const { return list_.rbegin(); }
    reverse_iterator rend() const { return list_.rend(); }

    size_t size() const { return list_.size(); }

    int operator[](const Def* def) const {
        PostOrderMap::const_iterator it = indices_.find(def);
        if(it != indices_.end())
            return it->second;
        return -1;
    }

    const Def* operator[](int postorder_index) const {
        if(postorder_index < 0 && postorder_index >= size())
            return 0;
        return list_[postorder_index];
    }

private:
    typedef boost::unordered_map<const Def*, int> PostOrderMap;

    bool visited(const Def* def) const;

    void init(const Def* def, int baseIndex);

    PostOrderMap indices_;
    PostOrderList list_;
};

}

#endif
