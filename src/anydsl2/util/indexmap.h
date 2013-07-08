#ifndef ANYDSL2_INDEXMAP_H
#define ANYDSL2_INDEXMAP_H

#include <vector>

namespace anydsl2 {

template<class T>
class IndexMap {
public:
    
    typedef typename std::vector<T*>::reverse_iterator reverse_iterator;
    typedef typename std::vector<T*>::const_reverse_iterator const_reverse_iterator;
    typedef typename std::vector<T*>::iterator iterator;
    typedef typename std::vector<T*>::const_iterator const_iterator;

    T*& operator [] (size_t handle) {
        if (handle >= vector_.size()) vector_.resize(handle + 1, (T*) 0);
        return vector_[handle];
    }
    const T*& operator [] (size_t handle) const { return (*static_cast<const IndexMap*>(this))[handle]; }
    T* find(size_t handle) const { return handle < vector_.size() ? vector_[handle] : 0; }
    bool empty() const { return vector_.empty(); }
    void clear() { vector_.clear(); }
    iterator begin() { return vector_.begin(); }
    iterator end()   { return vector_.end(); }
    const_iterator begin() const { return vector_.begin(); }
    const_iterator end()   const { return vector_.end(); }
    reverse_iterator rbegin() { return vector_.rbegin(); }
    reverse_iterator rend()   { return vector_.rend(); }
    const_reverse_iterator rbegin() const { return vector_.rbegin(); }
    const_reverse_iterator rend()   const { return vector_.rend(); }

private:

    std::vector<T*> vector_;
};

} // namespace anydsl2

#endif 
