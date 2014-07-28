CUDA_DIR   ?= $(shell dirname `which nvcc`)/..
INC_DIRS   = -I $(CUDA_DIR)/include \
             -I $(CUDA_DIR)/nvvm/include \
             -I $(CUDA_DIR)/nvvm/libnvvm-samples/common/include \
             -I .
LIB_DIRS   = $(CUDA_DIR)/lib64 $(CUDA_DIR)/nvvm/lib64
CXX_FLAGS := -Wall -Wunused
OPT_FLAGS := -O3
DBG_FLAGS :=
MAIN      ?= gaussian
RPATH_PRE := -Wl,-rpath,

OS:=$(shell uname -s)

ifeq ($(OS),Linux)
	LD_FLAGS = -lcuda -lnvvm
	SED_EXT =
endif
ifeq ($(OS),Darwin) # assume Mac OS X
	LD_FLAGS = -lcuda -lnvvm
	SED_EXT = ''
endif


all: $(MAIN).out
	@echo "Running $(MAIN) ..."
	@# remove attribute from asm
	@if [ -f $(MAIN).nvvm ]; then \
		sed -i $(SED_EXT) '/attributes #0/d' $(MAIN).nvvm; \
		sed -i $(SED_EXT) 's/#0//' $(MAIN).nvvm; \
	fi
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

%.out: %.s cu_runtime.s thorin_runtime.s thorin_utils.s
	clang++ $^ $(addprefix $(RPATH_PRE),$(LIB_DIRS)) $(addprefix -L ,$(LIB_DIRS)) $(LD_FLAGS) -o $@

clean:
	@rm -f $(MAIN).out *.ll *.o *.bc *.s

.PHONY: all

