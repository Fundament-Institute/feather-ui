.PHONY: all clean distclean

all:
	make -f feathergui.mk

clean:
	make clean -f feathergui.mk

dist: all distclean
	tar -czf feathergui-posix.tar.gz *

distclean:
	make distclean -f feathergui.mk

debug:
	make debug -f feathergui.mk
