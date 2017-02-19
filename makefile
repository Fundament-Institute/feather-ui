.PHONY: all clean distclean

all:
	make -f feathergui.mk
	make -f fgExample-c.mk

clean:
	make clean -f feathergui.mk
	make clean -f fgExample-c.mk

dist: all distclean
	tar -czf feathergui-posix.tar.gz *

distclean:
	make distclean -f feathergui.mk
	make distclean -f fgExample-c.mk

debug:
	make debug -f feathergui.mk
	make debug -f fgExample-c.mk
