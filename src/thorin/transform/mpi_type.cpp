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
            auto mem = entry->param(0);
            auto input = entry->param(1);
            auto output = entry->param(2);
            auto ret = entry->param(3);

            if(auto ptrType = input->type()->isa<PtrType>()) {
                //TODO adjust get_mpi_byte call with runtime call
                //generate continuations
                auto mpi_byte_call = world.continuation(world.fn_type({ world.mem_type(), world.fn_type({ world.mem_type(), world.type_qs32(1)})}), Debug(Symbol("get_mpi_byte")));
                mpi_byte_call->cc() = CC::C;
                auto mpi_byte_call_cont = world.continuation(world.fn_type({ world.mem_type(), world.type_qs32(1)}),Debug(Symbol("get_mpi_byte_cont")));

                auto mpi_type_contiguous_call = world.continuation(world.fn_type({
                    world.mem_type(),
                    world.type_qs32(1), //count
                    world.type_qs32(1), //oldtype
                    world.ptr_type(world.type_qs32(1)), //newtype
                    world.fn_type({ world.mem_type(), world.type_qs32(1) })}),Debug(Symbol("MPI_Type_contiguous")));
                mpi_byte_call->cc() = CC::C;
                auto mpi_type_contiguous_call_cont = world.continuation(world.fn_type({ world.mem_type(), world.type_qs32(1)}),Debug(Symbol("MPI_Type_contiguous_cont")));

                auto mpi_type_commit_call = world.continuation(world.fn_type({
                    world.mem_type(),
                    world.ptr_type(world.type_qs32()), //newtype
                    world.fn_type({ world.mem_type(), world.type_qs32(1)})}),Debug(Symbol("MPI_Type_commit")));
                mpi_type_commit_call->cc() = CC::C;
                auto mpi_type_commit_call_cont = world.continuation(world.fn_type({ world.mem_type(), world.type_qs32(1)}),Debug(Symbol("MPI_Type_commit_cont")));

                //create jumps
                entry->jump(mpi_byte_call, { mem, mpi_byte_call_cont });

                //preparations for mpi_type_contiguous
                auto inputSize = world.size_of(ptrType->pointee());

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
