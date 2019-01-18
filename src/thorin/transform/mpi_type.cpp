#include "thorin/transform/deep_copy.h"
#include "thorin/analyses/scope.h"
#include "thorin/world.h"
#include "deep_copy.h"

namespace thorin {

class MpiType {
public:
    MpiType(World& world)
        : world(world) {}

     void run() {
        Scope::for_each<false>(world, [&](Scope& scope) {
            auto entry = scope.entry();

            if (entry->intrinsic() != Intrinsic::MpiType)
                return;

            entry->intrinsic() = Intrinsic::None;

            //parameters:
            //0 - mem type
            //1 - struct input type
            //2 - mpi output datatype
            //3 - return function
            auto input = entry->param(1);
            bool invalidType = true;
            if(auto ptrType = input->type()->isa<PtrType>()) {
                if(auto structType = ptrType->pointee()->isa<StructType>()) {
                    invalidType = false;
                    std::vector<const Type*> params;
                    params.emplace_back(world.mem_type());
                    params.emplace_back(world.ptr_type(world.type_qs32(1),1));
                    for(size_t i=0; i<structType->num_ops(); i++) {
                        params.emplace_back(world.ptr_type(structType->op(i),1));
                    }
                    params.emplace_back(world.fn_type({world.mem_type()}));
                    auto externalCall = world.continuation(world.fn_type(params),Debug(Symbol("mpi_type_" + structType->name().str())));
                    externalCall->cc() = CC::C;

                    std::vector<const Def*> call_params;
                    call_params.emplace_back(entry->param(0));
                    call_params.emplace_back(entry->param(2));
                    for(size_t i=0; i<structType->num_ops(); i++) {
                        call_params.emplace_back(world.lea(entry->param(1), world.literal_qu32((qu32) i, {}), {}));
                    }
                    call_params.emplace_back(entry->param(3));

                    entry->jump(externalCall, call_params);
                }
            }
            if(invalidType) {
                std::cerr << "mpi_type is only available for struct types!" << std::endl;
                exit(1);
            }

            scope.update();
        });

        world.cleanup();
    }

private:
    World& world;
};
    void mpi_type(World& world) {
    MpiType(world).run();
}

}
