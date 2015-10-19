; Module thorin runtime decls
declare i8*  @thorin_alloc(i32, i32, i64);
declare void @thorin_release(i8*);
declare i8*  @thorin_map(i8*, i64, i64);
declare void @thorin_unmap(i8*);
declare void @thorin_copy(i8*, i8*);
declare void @thorin_set_block_size(i32, i32, i32, i32, i32);
declare void @thorin_set_grid_size(i32, i32, i32, i32, i32);
declare void @thorin_set_kernel_arg(i32, i32, i32, i8*, i32);
declare void @thorin_load_kernel(i32, i32, i8*, i8*);
declare void @thorin_launch_kernel(i32, i32);
declare void @thorin_synchronize(i32, i32);
declare void @thorin_parallel_for(i32, i32, i32, i8*, i8*);
declare i32  @thorin_spawn_thread(i8*, i8*);
declare void @thorin_sync_thread(i32, i32);

