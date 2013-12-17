#ifndef SPIR_DECL
#error SPIR_DECL not defined
#endif

SPIR_DECL(spir_malloc_buffer)
SPIR_DECL(spir_free_buffer)

SPIR_DECL(spir_write_buffer)
SPIR_DECL(spir_read_buffer)

SPIR_DECL(spir_build_program_and_kernel)
SPIR_DECL(spir_set_kernel_arg)
SPIR_DECL(spir_set_problem_size)

SPIR_DECL(spir_launch_kernel)
SPIR_DECL(spir_synchronize)

#undef SPIR_DECL
