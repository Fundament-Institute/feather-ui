CXX_FILES := $(notdir $(wildcard ./*.cpp))
C_FILES := $(notdir $(wildcard ./*.c))

CPPTEST_OBJDIR 		  := $(OBJDIR)/cpptest
CXX_OBJS          	  := $(foreach rule,$(CXX_FILES:.cpp=.o),$(CPPTEST_OBJDIR)/$(rule))
C_OBJS          	  := $(foreach rule,$(C_FILES:.c=.o),$(CPPTEST_OBJDIR)/$(rule))
CPPTEST_CPPFLAGS       := $(CPPFLAGS)
CPPTEST_DEBUG_CPPFLAGS := $(CPPFLAGS) -g3
LDFLAGS 			  := $(LDFLAGS) -lfgOpenGL -lGL
.PHONY: all clean

all: $(BINDIR)/cpptest
clean:
	$(RM) $(BINDIR)/cpptest
	$(RM) -r $(CPPTEST_OBJDIR)

$(BINDIR)/cpptest: $(CXX_OBJS) $(C_OBJS)
	@mkdir -p $(BINDIR)
	$(CXX) $(CXX_OBJS) $(C_OBJS) $(LDFLAGS) -o $@

$(CPPTEST_OBJDIR)/%.o: ./%.cpp
	@mkdir -p $(CPPTEST_OBJDIR)
	$(CXX) $(CPPTEST_CPPFLAGS) -MMD -c $< -o $@

$(CPPTEST_OBJDIR)/%.o: ./%.c
	@mkdir -p $(CPPTEST_OBJDIR)
	$(CXX) $(CPPTEST_CPPFLAGS) -MMD -c $< -o $@
