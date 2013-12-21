OPENCL_DIR ?= /opt/intel/opencl
INC_DIRS = -I $(OPENCL_DIR)/include
LIB_DIRS = -L $(OPENCL_DIR)/lib64
OPT_FLAGS := -O3
DBG_FLAGS :=

all: jacobi_generic.out
	@echo "Running main ..."
	@llvm-as simple-gpu64.spir
	@llvm-as a_kernel.spir
	@./jacobi_generic.out

%.llo: %.cpp cl_runtime.h
	clang++ $(DBG_FLAGS) $(INC_DIRS) -std=c++11 -o $@ -c -emit-llvm $<

#%.llo: %.cpp
#	clang++ $(DBG_FLAGS) $(INC_DIRS) -std=c++11 -o $@ -c -emit-llvm $<

%.bc: %.llo
	llvm-link $^ -o $*-unop.bc
	opt $*-unop.bc $(OPT_FLAGS) -o $@
	llvm-link $@ -o $@
	llvm-dis $@ -o $*.ll

%.bc: %.ll
	llvm-link $^ -o $*-unop.bc
	opt $*-unop.bc $(OPT_FLAGS) -o $@
	llvm-link $@ -o $@
	llvm-dis $@ -o $*.ll

%.s: %.bc
	llc $(OPT_FLAGS) $< -o $@

%.out: %.s cl_runtime.s
	clang++ $< cl_runtime.s $(LIB_DIRS) -lrt -lOpenCL -o $@

#%.out: %.s
#	clang++ $< $(LIB_DIRS) -lrt -lOpenCL -o $@

clean:
	@rm -f main $(MAIN_FILES) *.ll *.llo *.o *.bc *.s *.out

.PHONY: all

