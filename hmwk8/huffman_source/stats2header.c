/* -*-Mode: C;-*- */
/* $Id: stats2header.c 1.1 Wed, 05 Jun 1996 15:29:08 -0700 jmacd $	*/
/* huff.c: A utility for generating header files from tables. */

#include <sys/param.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include "huff.h"

#define ERROR_MSG_HEADER "stats2header: "

static int stats2header(const char* statstable,
			const char* headerfile,
			const char* description)
{
    FILE* outfile;
    int i, a;
    char buf[2];
    int TotalNodes;
    HuffStruct *H;

    outfile = fopen(statstable, "r");
    if(outfile == NULL) {
	perror(ERROR_MSG_HEADER"Failed opening stats file for read");
	return 0;
    }

    if(fscanf(outfile, "%d", &a) != 1) {
	fprintf(stderr, "Invalid stats file\n");
	return 0;
    }

    H = Huff_Initialize_Training_Encoder(a, statstable);

    TotalNodes = (2*H->AlphabetSize)-1;

    buf[1] = '\0';

    outfile = fopen(headerfile, "w");

    if(outfile == NULL) {
	perror(ERROR_MSG_HEADER"Failed opening header file for write");
	return 0;
    }

    fprintf(outfile,
"/* -*- C -*-\n"
" * Table of frequencies for Huffman compression trees.\n"
" * This file was automatically generated for the data\n"
" * described as:\n"
" *\n"
" *      %s\n"
" *\n"
" * Editing this file doesn't make a lot of sense, as this\n"
" * is an optimal tree for the particular set of data\n"
" * described above.\n"
" *\n"
" * Total element count: %d\n"
" * Alphabet size for this data: %d\n"
" */\n"
"\n"
"#define HN (HuffNode*)\n"
"\n"
"static HuffNode %s[%d] = {\n"
,
	    description,
	    H->RootNode->Weight,
	    H->AlphabetSize,
	    description,
	    TotalNodes);

    for(i = 0; i < H->AlphabetSize; i += 1) {
	buf[0] = i;
	fprintf(outfile, "    {%10d, HN %4d, 0, 0, 0, 0, 0},  "
		"/* element %d (%s) */\n",
		H->Alphabet[i].Weight,
		(int)(H->Alphabet[i].Parent - H->Alphabet),
		i, (i < 128 && isprint(i) ? buf : ""));
    }

    for(i = H->AlphabetSize; i < TotalNodes; i += 1) {
	buf[0] = i;
	fprintf(outfile, "    {%10d, HN %3d, HN %3d, HN %3d, 0, 0, 0}%s  "
		"/* element %d (%s) */\n",
		H->Alphabet[i].Weight,
		(i == H->AlphabetSize ? 0 : (int)(H->Alphabet[i].Parent - H->Alphabet)),
		(int)(H->Alphabet[i].LeftChild - H->Alphabet),
		(int)(H->Alphabet[i].RightChild - H->Alphabet),
		((i == (TotalNodes - 1)) ? "" : ","),
		i, (i < 128 && isprint(i) ? buf : ""));
    }

    fprintf(outfile, "\n};\n\n/* end of stats */\n");

    return 1;
}

int main(int argc, char** argv)
{
    if(argc != 4) {
	fprintf(stderr, "Usage: %s statsfile headerfile table_name\n", argv[0]);
	return 0;
    }

    return !stats2header(argv[1], argv[2], argv[3]);
}
