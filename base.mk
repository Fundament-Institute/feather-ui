C_OBJS := ${C_SRCS:.c=.o}
CXX_OBJS := ${CXX_SRCS:.cpp=.o}
OBJS := $(addprefix $(OBJDIR)/c/,$(C_OBJS)) $(addprefix $(OBJDIR)/cxx/,$(CXX_OBJS))
CPPFLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir)) -msse2 -mcx16 -Wl,-rpath -Wl,. 
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) -msse2 -mcx16 
LDLIBS := $(foreach library,$(LIBRARIES),-l$(library)) 

.PHONY: all clean distclean

all: $(BUILDDIR)/$(TARGET)
	
$(OBJDIR)/c/%.o : %.c
	+@[ -d $(OBJDIR) ] || mkdir -p $(OBJDIR)
	+@[ -d $(OBJDIR)/c ] || mkdir -p $(OBJDIR)/c
	+@[ -d $(OBJDIR)/c/$(SRCDIR) ] || mkdir -p $(OBJDIR)/c/$(SRCDIR)
	$(COMPILE.c) $< -o $@
          
$(OBJDIR)/cxx/%.o : %.cpp
	+@[ -d $(OBJDIR) ] || mkdir -p $(OBJDIR)
	+@[ -d $(OBJDIR)/cxx ] || mkdir -p $(OBJDIR)/cxx
	+@[ -d $(OBJDIR)/cxx/$(SRCDIR) ] || mkdir -p $(OBJDIR)/cxx/$(SRCDIR)
	$(COMPILE.cpp) $< -o $@
	
$(BUILDDIR)/$(TARGET): $(OBJS)
	+@[ -d $(BUILDDIR) ] || mkdir -p $(BUILDDIR)
	$(LINK.cc) $(OBJS) -o $@ $(LDLIBS)

clean: distclean
	@- $(RM) $(BUILDDIR)/$(TARGET)

debug: CPPFLAGS += -ggdb -fvar-tracking-assignments
debug: all
