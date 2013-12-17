#ifndef NVVM_DECL
#error NVVM_DECL not defined
#endif

NVVM_DECL(nvvm_malloc_memory)
NVVM_DECL(nvvm_free_memory)

NVVM_DECL(nvvm_write_memory)
NVVM_DECL(nvvm_read_memory)

NVVM_DECL(nvvm_load_kernel)
NVVM_DECL(nvvm_set_kernel_arg)
NVVM_DECL(nvvm_set_problem_size)

NVVM_DECL(nvvm_launch_kernel)
NVVM_DECL(nvvm_synchronize)

#undef NVVM_DECL
