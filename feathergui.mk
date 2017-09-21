TARGET := libfeathergui.so
SRCDIR := feathergui
BUILDDIR := bin
OBJDIR := bin/obj
C_SRCS := $(wildcard $(SRCDIR)/*.c)
CXX_SRCS := $(wildcard $(SRCDIR)/*.cpp)
INCLUDE_DIRS := include feathergui include/feathergui
LIBRARY_DIRS := 
LIBRARIES := rt

CPPFLAGS += -fPIC -D BSS_UTIL_EXPORTS -std=c++17 -Wall -Wno-attributes -Wno-unknown-pragmas -Wno-reorder -Wno-missing-braces -Wno-unused-function -Wno-comment -Wno-char-subscripts -Wno-switch -Wno-parentheses
LDFLAGS += -shared

include base.mk

distclean:
	@- $(RM) $(OBJS)
	@- $(RM) -r $(OBJDIR)

PREFIX = /usr

.PHONY: install
install:
	test -d $(PREFIX) || mkdir $(PREFIX)
	test -d $(PREFIX)/lib || mkdir $(PREFIX)/lib
	test -d $(PREFIX)/include || mkdir $(PREFIX)/include
	test -d $(PREFIX)/include/$(SRCDIR) || mkdir $(PREFIX)/include/$(SRCDIR)
	cp $(BUILDDIR)/$(TARGET) $(PREFIX)/lib/$(TARGET)
	cp include/$(SRCDIR)/*.h $(PREFIX)/include/$(SRCDIR)/

.PHONY: uninstall
uninstall:
	rm -f $(PREFIX)/lib/$(TARGET)
	rm -f -r $(PREFIX)/include/$(SRCDIR)