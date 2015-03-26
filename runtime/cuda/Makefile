CUDA_DIR  ?= $(shell dirname $(shell dirname `which nvcc`))
INC_DIRS   = -I $(CUDA_DIR)/include \
             -I $(CUDA_DIR)/nvvm/include \
             -I ../common -I .
CXX_FLAGS := -std=c++11 -Wall -Wunused
OPT_FLAGS := -O3
DBG_FLAGS :=
MAIN      ?= main
RPATH_PRE := -Wl,-rpath,
OS        := $(shell uname -s)
CPP        = cu_runtime.cpp ../common/thorin_runtime.cpp ../common/thorin_utils.cpp
OBJS       = $(CPP:%.cpp=%.o)

ifneq (,$(wildcard $(CUDA_DIR)/include/nvrtc.h))
	NVRTC_LIB := -lnvrtc
endif
LD_FLAGS = -lcuda -lnvvm $(NVRTC_LIB)
ifeq ($(OS),Linux)
	LIB_DIRS = $(CUDA_DIR)/lib64 $(CUDA_DIR)/nvvm/lib64
endif
ifeq ($(OS),Darwin) # assume Mac OS X
	LIB_DIRS = $(CUDA_DIR)/lib $(CUDA_DIR)/nvvm/lib
endif


all: $(MAIN).out
	@echo "Running $(MAIN) ..."
	@# remove attribute from asm
	@if [ -f $(MAIN).nvvm ]; then ../post-patcher.py nvvm $(MAIN); fi
	@CUDA_VISIBLE_DEVICES=0 ./$(MAIN).out
	@CUDA_VISIBLE_DEVICES=1 ./$(MAIN).out

%.o: %.cpp
	clang++ $(CXX_FLAGS) -DPROVIDE_MAIN -DNVCC_BIN='"$(CUDA_DIR)/bin/nvcc"' -DLIBDEVICE_DIR='"$(CUDA_DIR)/nvvm/libdevice/"' $(DBG_FLAGS) $(OPT_FLAGS) $(INC_DIRS) -c -o $@ $^

%.o: %.ll
	clang++ $(CXX_FLAGS) $(DBG_FLAGS) $(OPT_FLAGS) -c -o $@ $^

%.out: %.o $(OBJS)
	clang++ $^ $(addprefix $(RPATH_PRE),$(LIB_DIRS)) $(addprefix -L ,$(LIB_DIRS)) $(LD_FLAGS) -o $@

clean:
	@rm -f $(filter-out $(wildcard simple-gpu64.cu simple-gpu64.nvvm),$(wildcard *.ll *.o *.bc *.s *.cu *.nvvm *.ptx *.out))

.PHONY: all

