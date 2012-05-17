#include "anydsl/air/use.h"

#include "anydsl/air/def.h"

namespace anydsl {

Use::Use(Def* def, AIRNode* parent, const std::string& debug /*= ""*/)
    : AIRNode(Index_Use, debug)
    , def_(def) 
    , parent_(parent)
{
    def_->registerUse(this);
}

Use::~Use() {
    def_->unregisterUse(this);
}

//------------------------------------------------------------------------------

Args::Args(AIRNode* parent)
    : parent_(parent)
    , sentinel_(new Node())
    , size_(0)
{
    sentinel_->next_ = sentinel_;
    sentinel_->prev_ = sentinel_;
}

Args::~Args() {
    clear();
    delete sentinel_;
}

Args::iterator Args::insert(Args::iterator pos, Def* def, const std::string& debug /*= ""*/) {
    Node* newNode = new UseNode(def, parent_, debug);

    newNode->next_ = pos.n_;
    newNode->prev_ = pos.n_->prev_;

    newNode->next_->prev_ = newNode;
    newNode->prev_->next_ = newNode;

    ++size_;

    return iterator(newNode);
}

Args::iterator Args::erase(Args::iterator pos) {
    UseNode* n = (UseNode*) pos.n_;
    anydsl_assert(!n->isSentinel_, "this must not be the sentinel");

    iterator res(n->next_);
    n->next_->prev_ = n->prev_;
    n->prev_->next_ = n->next_;
    delete n;

    --size_;

    return res;
}

void Args::clear() {
    Node* i = head();

    while (i != sentinel_) {
        anydsl_assert(!i->isSentinel_, "this must not be the sentinel");
        UseNode* cur = (UseNode*) i;
        i = i->next_;
        delete cur;
    }

    size_ = 0;
    sentinel_->next_ = sentinel_;
    sentinel_->prev_ = sentinel_;
}

} // namespace anydsl
