TARGET := fgExample-c
SRCDIR := examples/c
BUILDDIR := bin
OBJDIR := bin/obj
C_SRCS := $(wildcard $(SRCDIR)/*.c)
CXX_SRCS := $(wildcard $(SRCDIR)/*.cpp)
INCLUDE_DIRS := include
LIBRARY_DIRS := 
LIBRARIES := feathergui rt pthread dl

CPPFLAGS += -w -std=gnu++11 -pthread
LDFLAGS += -L./bin/

include base.mk

distclean:
	@- $(RM) $(OBJS)
	@- $(RM) -r $(OBJDIR)
	@- $(RM) bin/*.txt
	@- $(RM) bin/*.ini
	@- $(RM) bin/*.json
	@- $(RM) bin/*.ubj
	@- $(RM) bin/*.xml

