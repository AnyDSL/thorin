CUDA_DIR  ?= /usr/local/cuda
INC_DIRS   = -I $(CUDA_DIR)/include \
             -I $(CUDA_DIR)/nvvm/include \
             -I $(CUDA_DIR)/nvvm/libnvvm-samples/common/include \
             -I ../common/
LIB_DIRS   = $(CUDA_DIR)/lib64 $(CUDA_DIR)/nvvm/lib64
CXX_FLAGS := -Wall -Wunused
OPT_FLAGS := -O3
DBG_FLAGS :=
MAIN      ?= gaussian
RPATH_PRE := -Wl,-rpath,

OS:=$(shell uname -s)

ifeq ($(OS),Linux)
	LD_FLAGS = -lcuda -lnvvm
endif
ifeq ($(OS),Darwin) # assume Mac OS X
	LD_FLAGS = -lcuda -lnvvm
endif


all: $(MAIN).out
	@echo "Running $(MAIN) ..."
	@./$(MAIN).out

%.s: %.cpp
	clang++ $(CXX_FLAGS) -DPROVIDE_MAIN -DNVCC_BIN='"$(CUDA_DIR)/bin/nvcc"' -DLIBDEVICE_DIR='"$(CUDA_DIR)/nvvm/libdevice/"' $(DBG_FLAGS) $(INC_DIRS) -std=c++11 -o $@ -S $^

%.bc: %.ll
	llvm-link $^ -o $*-unop.bc
	opt $*-unop.bc $(OPT_FLAGS) -o $@
	llvm-link $@ -o $@
	llvm-dis $@ -o $*.ll

%.s: %.bc
	llc $(OPT_FLAGS) $^ -o $@

%.out: %.s cu_runtime.s ../common/thorin_runtime.s
	clang++ $^ $(addprefix $(RPATH_PRE),$(LIB_DIRS)) $(addprefix -L ,$(LIB_DIRS)) $(LD_FLAGS) -o $@

clean:
	@rm -f $(MAIN).out *.ll *.o *.bc *.s

.PHONY: all

