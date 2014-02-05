OPENCL_DIR ?= /opt/intel/opencl
CXX_FLAGS := -Wall -Wunused
OPT_FLAGS := -O3
DBG_FLAGS :=

OS:=$(shell uname -s)

ifeq ($(OS),Linux)
	INC_DIRS = -I $(OPENCL_DIR)/include
	LIB_DIRS = -L $(OPENCL_DIR)/lib64
	LD_FLAGS = -lrt -lOpenCL
endif
ifeq ($(OS),Darwin) # Assume Mac Os X
	INC_DIRS =
	LIB_DIRS =
	LD_FLAGS = -framework OpenCL
endif


all: main.out
	@echo "Running main ..."
	@llvm-as simple-gpu64.spir
	@./main.out

%.llo: %.cpp cl_runtime.h
	clang++ $(CXX_FLAGS) $(DBG_FLAGS) $(INC_DIRS) -std=c++11 -o $@ -c -emit-llvm $<

#%.llo: %.cpp
#	clang++ $(CXX_FLAGS) $(DBG_FLAGS) $(INC_DIRS) -std=c++11 -o $@ -c -emit-llvm $<

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

#%.out: %.s cl_runtime.s
#	clang++ $< cl_runtime.s $(LIB_DIRS) $(LD_FLAGS) -o $@

%.out: %.s
	clang++ $< $(LIB_DIRS) $(LD_FLAGS) -o $@

clean:
	@rm -f main $(MAIN_FILES) *.ll *.llo *.o *.bc *.s *.out

.PHONY: all

