/* -*-Mode: C;-*- */
/* $Id: compact.c 1.1.1.1 Mon, 17 Jun 1996 18:47:03 -0700 jmacd $	*/
/* huff.c: A test for the huffman routines. */

#include <sys/param.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include "huff.h"
#include "stats.h"

#define ERROR_MSG_HEADER "compact: "

#ifndef GNUC
#define inline
#endif

static inline int read_byte(HuffStruct *decoder, FILE* out, unsigned int c)
{
    int i = 0;
    unsigned int shift = 1;
    int val;

    for(; i < 8; i += 1) {
	Bit b = (c & shift) && 1;
	val = Huff_Decode_Bit(decoder, b);

	if(val) {
	    int decode = Huff_Decode_Data(decoder);
	    if(decode == 256)
		return 1;
	    fputc(decode, out);
	}
	shift <<= 1;
    }
    return 0;
}

static inline void write_byte(HuffStruct *encoder, FILE* out, int c)
{
    int bits, i;
    static unsigned char data = 0, mask = 1;
    bits = Huff_Encode_Data(encoder, c);
/*    fprintf(stdout, "%d encodes as ", c); */
    for(i = 0; i < bits; i += 1) {
	if(Huff_Get_Encoded_Bit(encoder))
	    data |= mask;

/*	fprintf(stdout, "%d", data & mask && 1); */

	if(mask == 0x80) {
	    fputc(data, out);
	    mask = 1;
	    data = 0;
	} else {
	    mask <<= 1;
	}
    }
/*    fprintf(stdout, "\n"); */
}

int main(int argc, char** argv) {
    HuffStruct *encoder;
    const char* argv0;
    FILE* in, *out;
    int c;
    int compress;
    int use_fixed = 0;
    int use_train = 0;
    const char* train_file = NULL;
    char basename[266];

    if((argv0 = strrchr(argv[0], '/')) == NULL)
	argv0 = argv[0];
    else
	argv0 += 1;

    if(strcmp(argv0, "uncompact") == 0) {
	compress = 0;
    } else {
	compress = 1;
    }

    while(1) {
	if(argc > 1 && strcmp("-d", argv[1]) == 0) {
	    compress = 0;
	    argc -= 1;
	    argv += 1;
	} else if(argc > 1 && strcmp("-f", argv[1]) == 0) {
 	    use_fixed = 1;
	    use_train = 0;
	    argc -= 1;
	    argv += 1;
	} else if(argc > 2 && strcmp("-t", argv[1]) == 0) {
 	    use_fixed = 0;
	    use_train = 1;
	    train_file = argv[2];
	    argc -= 2;
	    argv += 2;
	} else {
	    break;
	}
    }

    if(argc != 2) {
	fprintf(stderr, "Usage: %s [-d] [-t training_table] [-f] filename\n", argv[0]);
	exit(1);
    }

    strcpy(basename, argv[1]);

    if(compress) {
	if(use_fixed)
	    encoder = Huff_Initialize_Fixed_Encoder(257, stats);
	else if(use_train)
	    encoder = Huff_Initialize_Training_Encoder(257, train_file);
	else
	    encoder= Huff_Initialize_Adaptive_Encoder(257);

	if(encoder == NULL) {
	    perror(ERROR_MSG_HEADER"Malloc failed");
	    exit(1);
	}

	strcat(basename, ".jz"); /* j for josh */

	in = fopen(argv[1], "r");
	out = fopen(basename, "w");

	if(in == NULL || out == NULL) {
	    perror(ERROR_MSG_HEADER"Fopen failed");
	    exit(1);
	}

	while((c = fgetc(in)) != EOF) {
	    write_byte(encoder, out, c);
	}

	write_byte(encoder, out, 256);

	fclose(in);
	fclose(out);
    } else {
	if(use_fixed)
	    encoder = Huff_Initialize_Fixed_Encoder(257, stats);
	else
	    encoder = Huff_Initialize_Adaptive_Encoder(257);

	if(encoder == NULL) {
	    perror(ERROR_MSG_HEADER"Malloc failed");
	    exit(1);
	}

	if(strlen(basename) > 3 &&
	   strcmp(basename + strlen(basename) - 3, ".jz") == 0) {
	    basename[strlen(basename) - 3] = '\0';
	} else {
	    strcat(basename, ".ujz");
	}

	in = fopen(argv[1], "r");
	out = fopen(basename, "w");

	if(in == NULL || out == NULL) {
	    perror(ERROR_MSG_HEADER"Fopen failed");
	    exit(1);
	}

	while((c = fgetc(in)) != EOF && !read_byte(encoder, out, c)) { }
    }

    if(use_train && Huff_Dump_Stats(encoder, train_file) != 0)
	exit(1);
    else
	exit(0);
}
