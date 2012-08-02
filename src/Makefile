CFG ?= default
-include config.$(CFG)

BUILDDIR ?= build/$(CFG)
BIN ?= $(BUILDDIR)/bin

LLVMCONFIG ?= llvm-config

CXXFLAGS += -I . -fPIC `$(LLVMCONFIG) --cxxflags` -fexceptions -frtti
LDFLAGS += -L $(BIN) `$(LLVMCONFIG) --ldflags`

ANYDSL_LIB  ?= anydsl
IMPALA_LIB  ?= impala
ANYDSL_APP_BIN  ?= $(BIN)/anydsl

ANYDSL_LIB_FILE  ?= $(BIN)/lib$(ANYDSL_LIB).a
IMPALA_LIB_FILE  ?= $(BIN)/lib$(IMPALA_LIB).a

ANYDSL_SRCS := $(wildcard anydsl/*.cpp anydsl/be/llvm/*.cpp anydsl/util/*.cpp)
IMPALA_SRCS := $(wildcard impala/*.cpp)
ANYDSL_APP_SRCS := $(wildcard app/*.cpp)

ANYDSL_OBJS :=  $(ANYDSL_SRCS:%.cpp=$(BUILDDIR)/%.o)
IMPALA_OBJS :=  $(IMPALA_SRCS:%.cpp=$(BUILDDIR)/%.o)
ANYDSL_APP_OBJS :=  $(ANYDSL_APP_SRCS:%.cpp=$(BUILDDIR)/%.o)

ANYDSL_DEPS :=  $(ANYDSL_OBJS:%.o=%.d)
IMPALA_DEPS :=  $(IMPALA_OBJS:%.o=%.d)
ANYDSL_APP_DEPS :=  $(ANYDSL_APP_OBJS:%.o=%.d)

Q ?= @

.SUFFIXES:
.PHONY: all clean distclean

all: $(ANYDSL_LIB_FILE) $(IMPALA_LIB_FILE) $(ANYDSL_APP_BIN)

DUMMY := $(shell mkdir -p $(BIN) $(sort $(dir $(ANYDSL_OBJS) $(IMPALA_OBJS) $(ANYDSL_APP_OBJS))))

-include  $(ANYDSL_DEPS)
-include  $(IMPALA_DEPS)
-include  $(ANYDSL_APP_DEPS)

$(ANYDSL_APP_BIN): $(ANYDSL_APP_OBJS) $(ANYDSL_LIB_FILE) $(IMPALA_LIB_FILE)
	@echo '===> LD $@'
	$(Q)$(CXX) $(ANYDSL_APP_OBJS) -l $(IMPALA_LIB) -l $(ANYDSL_LIB) `$(LLVMCONFIG) --libs` $(LDFLAGS) -o $@

$(ANYDSL_LIB_FILE): $(ANYDSL_OBJS)
	@echo '===> AR $@'
	$(Q)$(AR) rc $(ANYDSL_LIB_FILE) $(ANYDSL_OBJS)

$(IMPALA_LIB_FILE): $(IMPALA_OBJS)
	@echo '===> AR $@'
	$(Q)$(AR) rc $(IMPALA_LIB_FILE) $(IMPALA_OBJS)

$(BUILDDIR)/%.o: %.cpp
	@echo '===> CXX $<'
	$(Q)$(CXX) $(CXXFLAGS) -c -MMD $< -o $@

clean distclean:
	@echo '===> CLEAN'
	$(Q)rm -fr $(BUILDDIR)
