#ifndef THORIN_ANALYSES_SCHEDULE_H
#define THORIN_ANALYSES_SCHEDULE_H

#include "thorin/analyses/cfg.h"
#include "thorin/util/stream.h"

namespace thorin {

class PrimOp;

class Schedule : public Streamable {
public:
    enum Tag { Early, Late, Smart };

    class Block {
    public:
        Block(const Block&) = delete;
        Block& operator=(Block) = delete;

        Block() {}

        const CFNode* node() const { return node_; }
        Continuation* continuation() const { return node()->continuation(); }
        ArrayRef<const PrimOp*> primops() const { return primops_; }
        size_t index() const { return index_; }

        typedef ArrayRef<const PrimOp*>::const_iterator const_iterator;
        const_iterator begin() const { return primops().begin(); }
        const_iterator end() const { return primops().end(); }

    private:
        const CFNode* node_;
        std::vector<const PrimOp*> primops_;
        size_t index_;

        friend class Schedule;
        friend class Scheduler;
    };

    template<class Value>
    using Map = IndexMap<Schedule, const Block&, Value>;
    using Set = IndexSet<Schedule, const Block&>;

    Schedule(const Schedule&) = delete;
    Schedule& operator=(Schedule) = delete;

    Schedule(Schedule&& other)
        : scope_(std::move(other.scope_))
        , indices_(std::move(other.indices_))
        , blocks_(std::move(other.blocks_))
        , tag_(std::move(other.tag_))
    {}
    Schedule(const Scope&, Tag = Smart);

    Tag tag() const { return tag_; }
    const Scope& scope() const { return scope_; }
    const World& world() const { return scope().world(); }
    const CFA& cfa() const { return scope().cfa(); }
    const F_CFG& cfg() const { return scope().f_cfg(); }
    ArrayRef<Block> blocks() const { return blocks_; }
    size_t size() const { return blocks_.size(); }
    const Block& operator[](const CFNode* n) const { return blocks_[indices_[n]]; }
    static size_t index(const Block& block) { return block.index(); }
    void verify();

    // Note that we don't use overloading for the following methods in order to have them accessible from gdb.
    virtual std::ostream& stream(std::ostream&) const override;  ///< Streams thorin to file @p out.
    void write_thorin(const char* filename) const;               ///< Dumps thorin to file with name @p filename.
    void thorin() const;                                         ///< Dumps thorin to a file with an auto-generated file name.

    typedef ArrayRef<const Block>::const_iterator const_iterator;
    const_iterator begin() const { return blocks().begin(); }
    const_iterator end() const { return blocks().end(); }

    Block& operator[](const CFNode* n) { return blocks_[indices_[n]]; }
    void block_schedule();

    const Scope& scope_;
    F_CFG::Map<size_t> indices_;
    Array<Block> blocks_;
    Tag tag_;

private:
    friend class Scheduler;
};

inline Schedule schedule(const Scope& scope, Schedule::Tag tag = Schedule::Smart) { return Schedule(scope, tag); }
void verify_mem(World& );

}

#endif
