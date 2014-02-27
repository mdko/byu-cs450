CC = cc
#CFLAGS = -Wall -O3 -funroll-loops -DNDEBUG
#CFLAGS = -Wall -g
CFLAGS = -fsimple -xO5 -xchip=ultra

PURE_ARGS = -cache-dir=/home/nima0/zakhor/jmacd/pure -collector=/usr/sww/lib/gcc-lib/sparc-sun-solaris2.4/2.7.2/ld

PURIFY = purify $(PURE_ARGS)
QUANTIFY = quantify $(PURE_ARGS)

all: compact stats2header libhuff.a

libhuff.a: huff.o
	rm -f libhuff.a
	ar -cq libhuff.a huff.o

compact: libhuff.a compact.o
	$(CC) -o $@ compact.o -L. -lhuff

pure_compact: libhuff.a compact.o
	$(PURIFY) $(CC) -o $@ compact.o -L. -lhuff

stats2header: libhuff.a stats2header.o
	$(CC) -o $@ stats2header.o -L. -lhuff

clean:
	rm -f core *.o compact stats2header libhuff.a

depend:
	makedepend *.c

# DO NOT DELETE THIS LINE -- make depend depends on it.

compact.o: /usr/include/sys/param.h /usr/include/sys/types.h
compact.o: /usr/include/sys/feature_tests.h /usr/include/sys/isa_defs.h
compact.o: /usr/include/sys/machtypes.h /usr/include/sys/select.h
compact.o: /usr/include/sys/time.h /usr/include/sys/time.h
compact.o: /usr/include/sys/unistd.h /usr/include/stdlib.h
compact.o: /usr/include/stdio.h /usr/include/string.h /usr/include/ctype.h
compact.o: huff.h stats.h
huff.o: /usr/include/sys/param.h /usr/include/sys/types.h
huff.o: /usr/include/sys/feature_tests.h /usr/include/sys/isa_defs.h
huff.o: /usr/include/sys/machtypes.h /usr/include/sys/select.h
huff.o: /usr/include/sys/time.h /usr/include/sys/time.h
huff.o: /usr/include/sys/unistd.h /usr/include/stdlib.h /usr/include/stdio.h
huff.o: /usr/include/string.h /usr/include/ctype.h huff.h
stats2header.o: /usr/include/sys/param.h /usr/include/sys/types.h
stats2header.o: /usr/include/sys/feature_tests.h /usr/include/sys/isa_defs.h
stats2header.o: /usr/include/sys/machtypes.h /usr/include/sys/select.h
stats2header.o: /usr/include/sys/time.h /usr/include/sys/time.h
stats2header.o: /usr/include/sys/unistd.h /usr/include/stdlib.h
stats2header.o: /usr/include/stdio.h /usr/include/string.h
stats2header.o: /usr/include/ctype.h huff.h
