; Module spir-thorin runtime decls

declare i64 @spir_malloc_buffer(i32, i8*);
declare void @spir_free_buffer(i32, i64);

declare void @spir_write_buffer(i32, i64, i8*);
declare void @spir_read_buffer(i32, i64, i8*);

declare void @spir_set_problem_size(i32, i64, i64, i64);
declare void @spir_set_config_size(i32, i64, i64, i64);
declare void @spir_synchronize(i32);

declare void @spir_set_kernel_arg(i32, i8*, i64);
declare void @spir_set_kernel_arg_map(i32, i64);
declare void @spir_set_kernel_arg_struct(i32, i8*, i64);
declare void @spir_build_program_and_kernel_from_binary(i32, i8*, i8*);
declare void @spir_build_program_and_kernel_from_source(i32, i8*, i8*);
declare void @spir_launch_kernel(i32, i8*);
