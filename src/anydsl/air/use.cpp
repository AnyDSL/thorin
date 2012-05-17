#include "anydsl/air/use.h"

#include "anydsl/air/def.h"
#include "anydsl/air/terminator.h"

namespace anydsl {

Use::Use(AIRNode* parent, Def* def, const std::string& debug /*= ""*/)
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

Args::Args(Terminator* parent)
    : parent_(parent)
    , sentinel_(new Node())
    , size_(0)
{
    sentinel_->next_ = sentinel_;
    sentinel_->prev_ = sentinel_;
}

Args::~Args() {
    clear();
    anydsl_assert(sentinel_->isSentinel_, "this must be the sentinel");
    delete sentinel_;
}

Args::iterator Args::insert(Args::iterator pos, Def* def, const std::string& debug /*= ""*/) {
    Node* newNode = new UseNode(parent_, def, debug);
    Node* n = pos.n_;

    newNode->next_ = n;
    newNode->prev_ = n->prev_;

    n->prev_ = newNode;
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
