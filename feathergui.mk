TARGET := libfeathergui.so
SRCDIR := feathergui
BUILDDIR := bin
OBJDIR := bin/obj
C_SRCS := $(wildcard $(SRCDIR)/*.c)
CXX_SRCS := $(wildcard $(SRCDIR)/*.cpp)
INCLUDE_DIRS := include feathergui feathergui/bss-util
LIBRARY_DIRS := 
LIBRARIES := rt

CPPFLAGS += -fPIC -D BSS_UTIL_EXPORTS -std=gnu++0x -Wall -Wno-attributes -Wno-unknown-pragmas -Wno-reorder -Wno-missing-braces -Wno-unused-function -Wno-comment -Wno-char-subscripts
LDFLAGS += -shared

include base.mk

distclean:
	@- $(RM) $(OBJS)
	@- $(RM) -r $(OBJDIR)
