namespace thorin {
static const char* runtime_definitions = R"(
    ; Module anydsl runtime decls
    declare noalias [0 x i8]* @anydsl_alloc(i32, i64);
    declare noalias [0 x i8]* @anydsl_alloc_unified(i32, i64);
    declare void @anydsl_release(i32, [0 x i8]* noalias);
    declare void @anydsl_set_block_size(i32, i32, i32, i32);
    declare void @anydsl_set_grid_size(i32, i32, i32, i32);
    declare void @anydsl_set_kernel_arg(i32, i32, i8*, i32);
    declare void @anydsl_set_kernel_arg_ptr(i32, i32, i8*);
    declare void @anydsl_set_kernel_arg_struct(i32, i32, i8*, i32);
    declare void @anydsl_load_kernel(i32, i8*, i8*);
    declare void @anydsl_launch_kernel(i32);
    declare void @anydsl_synchronize(i32);
    declare void @anydsl_parallel_for(i32, i32, i32, i8*, i8*);
    declare i32  @anydsl_spawn_thread(i8*, i8*);
    declare void @anydsl_sync_thread(i32);
)";
}
