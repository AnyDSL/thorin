#ifndef THORIN_ANALYSES_DIVERGENCE_H
#define THORIN_ANALYSES_DIVERGENCE_H

#include "thorin/world.h"

namespace thorin {

class DivergenceAnalysis {
public:
    enum State {
        Varying,
        Uniform
    };

    GIDMap<Continuation*, ContinuationSet> dominatedBy;
    GIDMap<Continuation*, ContinuationSet> sinkedBy;
    GIDMap<Continuation*, ContinuationSet> loopBodies;
    GIDMap<Continuation*, ContinuationSet> loopExits;

    GIDMap<Continuation*, bool> isPredicated;

    GIDMap<Continuation*, ContinuationSet> relJoins;
    GIDMap<Continuation*, ContinuationSet> splitParrents;
private:
    Continuation *base;

    GIDMap<const Def*, State> uniform;

    void computeLoops();

    ContinuationSet successors(Continuation *cont);
    ContinuationSet predecessors(Continuation *cont);

public:
    DivergenceAnalysis(Continuation* base) : base(base) {};
    void run();
    State getUniform(const Def * def);

    void dump();
};

}

#endif
