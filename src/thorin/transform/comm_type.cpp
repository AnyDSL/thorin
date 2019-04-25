#include "thorin/transform/deep_copy.h"
#include "thorin/analyses/scope.h"
#include "thorin/world.h"
#include "deep_copy.h"

namespace thorin {

class CommType {
public:
    CommType(World& world)
        : world(world) {}

     void run() {
        Scope::for_each<false>(world, [&](Scope& scope) {
            auto entry = scope.entry();

            if (entry->intrinsic() != Intrinsic::CommType)
                return;

            entry->intrinsic() = Intrinsic::None;

            //parameters:
            //0 - mem type
            //1 - input type
            //2 - return function
            const Def* mem = entry->param(0);
            auto input = entry->param(1);
            auto ret = entry->param(2);
            const Def* inputSize;

            if(auto ptrType = input->type()->isa<PtrType>()) {
                inputSize = world.size_of(ptrType->pointee());
                if(auto structType = ptrType->pointee()->isa<StructType>()) {
                    if(structType->name().str() == "Buffer" && structType->num_ops() == 3) {
                        //load size from Buffer struct and cast it to i32
                        auto sizePtr = world.lea(input, world.literal_qu32(1, {}), {});
                        auto sizePtrLoaded = world.load(mem, sizePtr);
                        mem = world.extract(sizePtrLoaded, (u32) 0);
                        inputSize = world.cast(world.type_qs32(1),world.extract(sizePtrLoaded, (u32) 1));
                    }
                }

                auto create_datatype_call = world.continuation(world.fn_type({
                    world.mem_type(),
                    world.type_qs32(1),
                    world.fn_type({
                        world.mem_type(),
                        world.ptr_type(world.type_qs32(1))
                    })
                }), Debug(Symbol("anydsl_create_datatype")));
                create_datatype_call->cc() = CC::C;

                entry->jump(create_datatype_call, { mem, inputSize, ret });
            }
            else {
                std::cerr << "Invalid datatype in comm_type call!" << std::endl;
            }

            scope.update();
        });

        world.cleanup();
    }

private:
    World& world;
};
    void comm_type(World& world) {
    CommType(world).run();
}

}
