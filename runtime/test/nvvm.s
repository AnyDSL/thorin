; Module nvvm-thorin runtime decls

declare i64 @nvvm_malloc_memory(i32, i8*);
declare void @nvvm_free_memory(i32, i64);

declare void @nvvm_write_memory(i32, i64, i8*);
declare void @nvvm_read_memory(i32, i64, i8*);

declare void @nvvm_set_problem_size(i32, i64, i64, i64);
declare void @nvvm_set_config_size(i32, i64, i64, i64);
declare void @nvvm_synchronize(i32);

declare void @nvvm_set_kernel_arg(i32, i64*);
declare void @nvvm_set_mapped_kernel_arg(i32, i8*);
declare void @nvvm_load_kernel(i32, i8*, i8*);
declare void @nvvm_launch_kernel(i32, i8*);
