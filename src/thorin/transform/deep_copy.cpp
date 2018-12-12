#include "thorin/transform/deep_copy.h"
#include "thorin/analyses/scope.h"
#include "thorin/world.h"
#include "deep_copy.h"

namespace thorin {

class DeepCopy {
public:
    DeepCopy(World& world)
        : world(world) {}

     void run() {
        Scope::for_each<false>(world, [&](Scope& scope) {
            auto entry = scope.entry();

            if (entry->intrinsic() != Intrinsic::DeepCopy)
                return;
            auto type = entry->param(1)->type()->as<PtrType>()->pointee();
            add_continuation(type, entry);
            if(type2cont.find(type) == type2cont.end()) {
                type2cont[type] = entry;
            }
            entry->intrinsic() = Intrinsic::None;

            Continuation* call_cont = emit(type);
            if(call_cont != entry) {
                entry->jump(call_cont, entry->params_as_defs());
            }
            scope.update();
        });

        world.cleanup();
    }

    Continuation* emit(const Type* type) {
        auto res = type2cont.find(type);

        if(res != type2cont.end() && !res->second->empty()) {
            //Continuation has already been generated!
            return res->second;
        }
        else {
            //Continuation must be generated!
            if(res == type2cont.end()) {
                return nullptr;
            }

            Continuation* cont = res->second;

            auto input = cont->param(1);
            auto output = cont->param(2);
            auto return_fn = cont->param(3);

            auto frame = world.enter(cont->param(0));
            auto mem = world.extract(frame, (qu32) 0);

            const StructType* struct_type;
            const TupleType* tuple_type;

            if(auto prim_type = type->isa<PrimType>()) {
                //load input
                auto inputDef = world.load(mem, input);
                auto inputValue = world.extract(inputDef, (qu32) 1);
                mem = world.extract(inputDef, (qu32) 0);
                auto store = world.store(mem, output, inputValue);

                cont->jump(return_fn, { store });

                return cont;
            }
            else if(auto pointer_type = type->isa<PtrType>()) {
                auto pointee = pointer_type->pointee();

                //load input pointer
                auto inputDef = world.load(mem, input);
                auto inputValue = world.extract(inputDef, (qu32) 1);
                mem = world.extract(inputDef, (qu32) 0);
                //load output pointer
                auto outputDef = world.load(mem, output);
                auto outputValue = world.extract(outputDef, (qu32) 1);
                mem = world.extract(outputDef, (qu32) 0);

                check_continuation_availability(pointee);

                auto pointee_cont = emit(pointee);
                cont->jump(pointee_cont, { mem, inputValue, outputValue, return_fn});

                return cont;
            }
            else if((struct_type = type->isa<StructType>()) || (tuple_type = type->isa<TupleType>())) {
                size_t dimension = 0;
                std::string debug_name;

                if(struct_type) {
                    dimension = struct_type->num_ops();
                    debug_name = "deep_copy_struct_cont";
                    if(dimension == 3 && struct_type->name().str() == "Buffer" ) {
                        //special case: Buffer
                        auto sizePtr = world.lea(input, world.literal_qu32((qu32) 1, {}), {});
                        auto inputArrayPtrPtr = world.lea(input, world.literal_qu32((qu32) 0, {}), {});
                        auto outputArrayPtrPtr = world.lea(output, world.literal_qu32((qu32) 0, {}), {});
                        //load input and output array ptr ptr's
                        auto inputArrayPtr = world.load(mem, inputArrayPtrPtr);
                        mem = world.extract(inputArrayPtr, world.literal_qu32((qu32) 0, {}), {});
                        inputArrayPtr = world.extract(inputArrayPtr, world.literal_qu32((qu32) 1, {}), {});
                        auto outputArrayPtr = world.load(mem, outputArrayPtrPtr);
                        mem = world.extract(outputArrayPtr, world.literal_qu32((qu32) 0, {}), {});
                        outputArrayPtr = world.extract(outputArrayPtr, world.literal_qu32((qu32) 1, {}), {});
                        //load size and cast it to qs32
                        auto size = world.load(mem, sizePtr);
                        mem = world.extract(size, world.literal_qu32((qu32) 0, {}), {});
                        size = world.extract(size, world.literal_qu32((qu32) 1, {}), {});
                        size = world.cast(world.type_qs32(1), size, {});

                        auto loop = world.continuation(world.fn_type({world.mem_type(), world.type_qs32(1)}),Debug(Symbol("loop")));
                        cont->jump(loop, {mem, world.literal_qs32((qs32) 0, {})});

                        auto loop_mem = loop->param(0);
                        auto loop_index = loop->param(1);

                        //generate continuations of loop branch
                        auto if_then = world.continuation(world.fn_type(), Debug(Symbol("if_then")));
                        auto if_else = world.continuation(world.fn_type(), Debug(Symbol("if_else")));

                        //generate condition of loop
                        auto condition = world.cmp_lt(loop_index, size);
                        loop->branch(condition, if_then, if_else);

                        //fill branch continuations
                        //load input
                        auto inputPtr = if_then->world().lea(inputArrayPtr, loop_index, {});
                        auto input_value = if_then->world().load(loop_mem, inputPtr);
                        auto if_then_mem = if_then->world().extract(input_value, world.literal_qu32((qu32) 0, {}), {});
                        input_value = if_then->world().extract(input_value, world.literal_qu32((qu32) 1, {}), {});
                        //load output pointer
                        auto outputPtr = if_then->world().lea(outputArrayPtr, loop_index, {});
                        //store loaded value
                        if_then_mem = if_then->world().store(if_then_mem, outputPtr, input_value, {});
                        //increse loop counter
                        auto new_loop_index = world.arithop_add(world.literal_qs32((qs32) 1, {}), loop_index,{});
                        if_then->jump(loop, {if_then_mem, new_loop_index});

                        if_else->jump(return_fn, {loop_mem});

                        return cont;
                    }
                }
                else {
                    dimension = tuple_type->num_ops();
                    debug_name = "deep_copy_tuple_cont";
                }

                if(dimension > 0) {
                    //load input and output pointers
                    const Def* inputs[dimension];
                    const Def* outputs[dimension];
                    Continuation* calls[dimension];

                    for(u64 i=0; i<dimension; i++) {
                        inputs[i] = world.lea(input, world.literal_qu32((qu32) i, {}), {});
                        outputs[i] = world.lea(output, world.literal_qu32((qu32) i, {}), {});
                    }

                    for(u64 i=0; i<dimension; i++) {
                        if(auto cur_type = inputs[i]->type()->isa<PtrType>()) {
                            check_continuation_availability(cur_type->pointee());
                            calls[i] = emit(cur_type->pointee());
                        }
                        else {
                            std::cerr << "ERROR: expected pointer type as result of lea instruction!" << std::endl;
                            exit(1);
                        }
                    }

                    Continuation* call_continuations[dimension-1];
                    for(u64 i=0; i < dimension-1; i++) {
                        call_continuations[i] = world.continuation(world.fn_type({world.mem_type()}),Debug(Symbol(debug_name)));
                    }

                    auto cur_cont = cont;
                    for(u64 i=0; i < dimension; i++) {
                        //get current continuation
                        if(i > 0) {
                            cur_cont = call_continuations[i-1];
                            mem = cur_cont->param(0);
                        }

                        if(i == dimension - 1) {
                            cur_cont->jump(calls[i], {mem, inputs[i], outputs[i], return_fn});
                        }
                        else {
                            cur_cont->jump(calls[i], {mem, inputs[i], outputs[i], call_continuations[i]});
                        }
                    }
                }
                else {
                    //special case ()
                    cont->jump(return_fn, {mem});
                }

                return cont;
            }
            else if(auto array_type = type->isa<DefiniteArrayType>()) {
                auto elem_type = array_type->elem_type();
                auto dimension = array_type->dim();

                //load input and output pointers
                const Def* inputs[dimension];
                const Def* outputs[dimension];

                for(u64 i=0; i<dimension; i++) {
                    inputs[i] = world.lea(input, world.literal_qu32((qu32) i, {}), {});
                    outputs[i] = world.lea(output, world.literal_qu32((qu32) i, {}), {});
                }

                check_continuation_availability(elem_type);
                auto call = emit(elem_type);

                Continuation* call_continuations[dimension-1];
                for(u64 i=0; i < dimension-1; i++) {
                    call_continuations[i] = world.continuation(world.fn_type({world.mem_type()}),Debug(Symbol("deep_copy_array_cont")));
                }

                auto cur_cont = cont;
                for(u64 i=0; i < dimension; i++) {
                    //get current continuation
                    if(i > 0) {
                        cur_cont = call_continuations[i-1];
                        mem = cur_cont->param(0);
                    }

                    if(i == dimension - 1) {
                        cur_cont->jump(call, {mem, inputs[i], outputs[i], return_fn});
                    }
                    else {
                        cur_cont->jump(call, {mem, inputs[i], outputs[i], call_continuations[i]});
                    }
                }

                return cont;
            }
            else {
                std::cerr << "Trying to generate deep_copy of invalid type " << type->to_string() << "!" << std::endl;
                exit(1);
            }
        }
    }
private:
    World& world;
    TypeMap<Continuation*> type2cont;
    void add_continuation(const Type* type, Continuation* cont) {
        if(type2cont.find(type) == type2cont.end()) {
            type2cont[type] = cont;
        }
    }
    Continuation* generate_continuation(const Type* type) {
        auto ptr_type = world.ptr_type(type, 1);
        return world.continuation(world.fn_type({ world.mem_type(), ptr_type, ptr_type, world.fn_type({ world.mem_type() }) }),Debug(Symbol("deep_copy")));
    }

    //checks if there exists already a deep_copy continuation for the given type
    //@returns true iff a new continuation has been generated
    bool check_continuation_availability(const Type* type) {
        if(type2cont.find(type) == type2cont.end()) {
            //new continuation must be generated
            auto new_cont = generate_continuation(type);
            add_continuation(type, new_cont);
            return true;
        }
        return false;
    }
};

void deep_copy(World& world) {
    DeepCopy(world).run();
}

}
