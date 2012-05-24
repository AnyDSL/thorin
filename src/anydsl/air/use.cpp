#include "anydsl/air/use.h"

#include "anydsl/air/def.h"
#include "anydsl/air/jump.h"

namespace anydsl {

Use::Use(AIRNode* parent, Def* def)
    : AIRNode(Index_Use)
    , def_(def) 
    , parent_(parent)
{
    def_->registerUse(this);
}

Use::~Use() {
    def_->unregisterUse(this);
}

World& Use::world() {
    return def_->world();
}

//------------------------------------------------------------------------------

Ops::Ops()
    : sentinel_(new Node())
    , size_(0)
{
    sentinel_->next_ = sentinel_;
    sentinel_->prev_ = sentinel_;
}

Ops::~Ops() {
    clear();
    anydsl_assert(sentinel_->isSentinel_, "this must be the sentinel");
    delete sentinel_;
}

Ops::iterator Ops::insert(Ops::iterator pos, Def* parent, Def* def) {
    Node* newNode = new UseNode(parent, def);
    Node* n = pos.n_;

    newNode->next_ = n;
    newNode->prev_ = n->prev_;

    n->prev_ = newNode;
    newNode->prev_->next_ = newNode;

    ++size_;

    return iterator(newNode);
}

Ops::iterator Ops::erase(Ops::iterator pos) {
    UseNode* n = (UseNode*) pos.n_;
    anydsl_assert(!n->isSentinel_, "this must not be the sentinel");

    iterator res(n->next_);
    n->next_->prev_ = n->prev_;
    n->prev_->next_ = n->next_;
    delete n;

    --size_;

    return res;
}

void Ops::clear() {
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
