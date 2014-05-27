CUDA_DIR  ?= /usr/local/cuda
INC_DIRS   = -I $(CUDA_DIR)/include \
             -I $(CUDA_DIR)/nvvm/include \
             -I $(CUDA_DIR)/nvvm/libnvvm-samples/common/include
LIB_DIRS   = -L $(CUDA_DIR)/nvvm/lib64
CXX_FLAGS := -Wall -Wunused
OPT_FLAGS := -O3
DBG_FLAGS :=
MAIN      := gaussian

OS:=$(shell uname -s)

ifeq ($(OS),Linux)
	LD_FLAGS = -lcuda -lnvvm
endif
ifeq ($(OS),Darwin) # assume Mac OS X
	LD_FLAGS = -lcuda -lnvvm
endif


all: $(MAIN).out
	@echo "Running $(MAIN) ..."
	@ln -sf $(CUDA_DIR)/nvvm/libdevice/* .
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

%.out: %.s cu_runtime.s
	clang++ $< cu_runtime.s $(LIB_DIRS) $(LD_FLAGS) -o $@

clean:
	@rm -f $(MAIN).out *.ll *.o *.bc *.s

.PHONY: all

