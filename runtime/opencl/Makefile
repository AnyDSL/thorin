INC_DIRS   = -I .
CXX_FLAGS := -std=c++11 -Wall -Wunused
OPT_FLAGS := -O3
DBG_FLAGS :=
MAIN      ?= gaussian
OS        :=$(shell uname -s)
CPP        = $(shell echo thorin_*.cpp cl_*.cpp)
OBJS       = $(CPP:%.cpp=%.o)

ifeq ($(OS),Linux)
	OPENCL_DIR ?= /opt/intel/opencl
	INC_DIRS   += -I $(OPENCL_DIR)/include
	LIB_DIRS    = -L $(OPENCL_DIR)/lib64
	LD_FLAGS    = -lrt -lOpenCL
endif
ifeq ($(OS),Darwin) # assume Mac OS X
	LIB_DIRS    =
	LD_FLAGS    = -framework OpenCL
endif


all: $(MAIN).out
	@echo "Running $(MAIN) ..."
	@if [ -f $(MAIN).spir ]; then llvm-as $(MAIN).spir; fi
	@./$(MAIN).out

%.o: %.cpp
	clang++ $(CXX_FLAGS) -DPROVIDE_MAIN $(DBG_FLAGS) $(OPT_FLAGS) $(INC_DIRS) -c -o $@ $^

%.o: %.ll
	clang++ $(CXX_FLAGS) $(DBG_FLAGS) $(OPT_FLAGS) -c -o $@ $^

%.out: %.o $(OBJS)
	clang++ $^ $(LIB_DIRS) $(LD_FLAGS) -o $@

clean:
	@rm -f $(filter-out $(wildcard main.cl main.spir),$(wildcard *.ll *.o *.bc *.s *.cl *.spir *.out))

.PHONY: all

