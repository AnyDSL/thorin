OPENCL_DIR ?= /opt/intel/opencl
CXX_FLAGS := -Wall -Wunused
OPT_FLAGS := -O3
DBG_FLAGS :=
MAIN      := jacobi

OS:=$(shell uname -s)

ifeq ($(OS),Linux)
	INC_DIRS = -I $(OPENCL_DIR)/include
	LIB_DIRS = -L $(OPENCL_DIR)/lib64
	LD_FLAGS = -lrt -lOpenCL
endif
ifeq ($(OS),Darwin) # assume Mac OS X
	INC_DIRS =
	LIB_DIRS =
	LD_FLAGS = -framework OpenCL
endif


all: $(MAIN).out
	@echo "Running $(MAIN) ..."
	@if [ -f $(MAIN).spir ]; then llvm-as $(MAIN).spir; fi
	@./$(MAIN).out

%.s: %.cpp
	clang++ $(CXX_FLAGS) $(DBG_FLAGS) $(INC_DIRS) -std=c++11 -o $@ -S $<

%.bc: %.ll
	llvm-link $^ -o $*-unop.bc
	opt $*-unop.bc $(OPT_FLAGS) -o $@
	llvm-link $@ -o $@
	llvm-dis $@ -o $*.ll

%.s: %.bc
	llc $(OPT_FLAGS) $< -o $@

%.out: %.s cl_runtime.s
	clang++ $< cl_runtime.s $(LIB_DIRS) $(LD_FLAGS) -o $@

clean:
	@rm -f $(MAIN).out *.ll *.o *.bc *.s

.PHONY: all

