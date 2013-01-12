#ifndef ANYDSL2_INDEXMAP_H
#define ANYDSL2_INDEXMAP_H

#include "anydsl2/util/for_all.h"

#include <vector>

namespace anydsl2 {

template<class T>
class IndexMap {
public:
    
    ~IndexMap() {
        for (size_t i = 0, e = vector_.size(); i != e; ++i)
            delete vector_[i];
    }

    T*& operator [] (size_t handle) {
        if (handle >= vector_.size()) vector_.resize(handle + 1, 0);
        return vector_[handle];
    }

    const T*& operator [] (size_t handle) const {
        if (handle >= vector_.size()) vector_.resize(handle + 1, 0);
        return vector_[handle];
    }

    bool empty() const { return vector_.empty(); }

    T* find(size_t handle) const { return handle < vector_.size() ? vector_[handle] : 0; }

private:

    std::vector<T*> vector_;
};

} // namespace anydsl2

#endif 
