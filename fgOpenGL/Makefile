CXX_FILES := $(notdir $(wildcard ./*.cpp))
C_FILES := $(notdir $(wildcard ./*.c))
CXXLD    ?= $(CXX)

OPENGL_OBJDIR 		  := $(OBJDIR)/fgOpenGL
CXX_OBJS          	  := $(foreach rule,$(CXX_FILES:.cpp=.o),$(OPENGL_OBJDIR)/$(rule))
C_OBJS          	  := $(foreach rule,$(C_FILES:.c=.o),$(OPENGL_OBJDIR)/$(rule))
OPENGL_CPPFLAGS       := $(CPPFLAGS) -fPIC
OPENGL_DEBUG_CPPFLAGS := $(CPPFLAGS) -g3 -fPIC

.PHONY: all clean

all: $(LIBDIR)/libfgOpenGL.so
clean:
	$(RM) $(LIBDIR)/libfgOpenGL.so
	$(RM) -r $(OPENGL_OBJDIR)

$(LIBDIR)/libfgOpenGL.so: $(CXX_OBJS) $(C_OBJS)
	@mkdir -p $(LIBDIR)
	$(CXXLD) $(INNATIVE_OBJS) $(LDFLAGS) -shared -o $@

$(OPENGL_OBJDIR)/%.o: ./%.cpp
	@mkdir -p $(OPENGL_OBJDIR)
	$(CXX) $(OPENGL_CPPFLAGS) -MMD -c $< -o $@

$(OPENGL_OBJDIR)/%.o: ./%.c
	@mkdir -p $(OPENGL_OBJDIR)
	$(CXX) $(OPENGL_CPPFLAGS) -MMD -c $< -o $@
