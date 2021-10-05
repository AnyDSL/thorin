
#ifndef THORIN_CLOSURE_CONV_H
#define THORIN_CLOSURE_CONV_H

#include "thorin/world.h"

namespace thorin {

    class ClosureConv {
        public:
            ClosureConv(World& world)
                : world_(world)
                , closures_(DefMap<Closure>())
                , closure_types_(Def2Def())
                , worklist_(std::queue<const Def*>()) {};

            void run();
        private:
            
            const Def* rewrite(const Def *old_def, Def2Def *subst = nullptr);

            const Def* closure_type(const Pi *pi, const Def *ent_type = nullptr);

            struct Closure {
                Lam *old_fn;
                const Def *env;
                Lam *fn;
            };


            Closure make_closure(Lam *lam);

            World& world() { return world_; }

            World& world_;
            DefMap<Closure> closures_;
            Def2Def closure_types_;
            std::queue<const Def*> worklist_;
    };

};

#endif
