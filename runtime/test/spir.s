; Module spir-thorin runtime decls

declare i64 @spir_malloc_buffer(i8*);
declare void @spir_free_buffer(i64);

declare void @spir_write_buffer(i64, i8*);
declare void @spir_read_buffer(i64, i8*);

declare void @spir_set_problem_size(i64, i64, i64);
declare void @spir_set_config_size(i64, i64, i64);
declare void @spir_synchronize();

declare void @spir_set_kernel_arg(i64*, i64);
declare void @spir_set_mapped_kernel_arg(i8*);
declare void @spir_build_program_and_kernel_from_binary(i8*, i8*);
declare void @spir_build_program_and_kernel_from_source(i8*, i8*);
declare void @spir_launch_kernel(i8*);
