INC_DIRS   = -I .
CXX_FLAGS := -std=c++11 -Wall -Wunused
OPT_FLAGS := -O3
DBG_FLAGS :=
MAIN      ?= gaussian
RPATH_PRE := -Wl,-rpath,

OS:=$(shell uname -s)

ifeq ($(OS),Linux)
	LD_FLAGS = -lrt
endif
ifeq ($(OS),Darwin) # assume Mac OS X
	LD_FLAGS =
endif

CPP = $(shell echo *.cpp)
OBJS = $(CPP:%.cpp=%.o)


all: $(MAIN).out
	@echo "Running $(MAIN) ..."
	@./$(MAIN).out

%.o: %.cpp
	clang++ $(CXX_FLAGS) -DPROVIDE_MAIN $(DBG_FLAGS) $(OPT_FLAGS) $(INC_DIRS) -c -o $@ $^

%.o: %.ll
	clang++ $(CXX_FLAGS) $(DBG_FLAGS) $(OPT_FLAGS) -c -o $@ $^

%.out: %.o $(OBJS)
	clang++ $^ $(LD_FLAGS) -o $@

clean:
	@rm -f $(MAIN).out *.ll *.o

.PHONY: all

