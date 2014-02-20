; Module nvvm-thorin runtime decls

declare i64 @nvvm_malloc_memory(i8*);
declare void @nvvm_free_memory(i64);

declare void @nvvm_write_memory(i64, i8*);
declare void @nvvm_read_memory(i64, i8*);

declare void @nvvm_set_problem_size(i64, i64, i64);
declare void @nvvm_set_config_size(i64, i64, i64);
declare void @nvvm_synchronize();

declare void @nvvm_set_kernel_arg(i64*);
declare void @nvvm_load_kernel(i8*, i8*);
declare void @nvvm_launch_kernel(i8*);
