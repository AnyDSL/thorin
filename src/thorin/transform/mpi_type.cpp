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
            //1 - input type
            //2 - mpi output datatype
            //3 - return function
            const Def* mem = entry->param(0);
            auto input = entry->param(1);
            auto output = entry->param(2);
            auto ret = entry->param(3);
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
                //TODO adjust get_mpi_byte call with runtime call
                //generate continuations
                auto mpi_byte_call = world.continuation(world.fn_type({ world.mem_type(), world.fn_type({ world.mem_type(), world.type_qs32(1)})}), Debug(Symbol("anydsl_comm_get_byte")));
                mpi_byte_call->cc() = CC::C;
                auto mpi_byte_call_cont = world.continuation(world.fn_type({ world.mem_type(), world.type_qs32(1)}),Debug(Symbol("get_mpi_byte_cont")));

                auto mpi_type_contiguous_call = world.continuation(world.fn_type({
                    world.mem_type(),
                    world.type_qs32(1), //count
                    world.type_qs32(1), //oldtype
                    world.ptr_type(world.type_qs32(1)), //newtype
                    world.fn_type({ world.mem_type(), world.type_qs32(1) })}),Debug(Symbol("anydsl_comm_type_contiguous")));
                mpi_byte_call->cc() = CC::C;
                auto mpi_type_contiguous_call_cont = world.continuation(world.fn_type({ world.mem_type(), world.type_qs32(1)}),Debug(Symbol("type_contiguous_cont")));

                auto mpi_type_commit_call = world.continuation(world.fn_type({
                    world.mem_type(),
                    world.ptr_type(world.type_qs32()), //newtype
                    world.fn_type({ world.mem_type(), world.type_qs32(1)})}),Debug(Symbol("anydsl_comm_type_commit")));
                mpi_type_commit_call->cc() = CC::C;
                auto mpi_type_commit_call_cont = world.continuation(world.fn_type({ world.mem_type(), world.type_qs32(1)}),Debug(Symbol("type_commit_cont")));

                //create jumps
                entry->jump(mpi_byte_call, { mem, mpi_byte_call_cont });

                mpi_byte_call_cont->jump(mpi_type_contiguous_call, {
                    mpi_byte_call_cont->param(0), //mem
                    inputSize, //count
                    mpi_byte_call_cont->param(1), //oldtype
                    output, //newtype
                    mpi_type_contiguous_call_cont
                });

                mpi_type_contiguous_call_cont->jump(mpi_type_commit_call, {
                    mpi_type_contiguous_call_cont->param(0), //mem
                    output, //newtype
                    mpi_type_commit_call_cont
                });

                mpi_type_commit_call_cont->jump(ret, { mpi_type_commit_call_cont->param(0)});

            }
            else {
                std::cerr << "Invalid datatype in mpi_type call!" << std::endl;
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
