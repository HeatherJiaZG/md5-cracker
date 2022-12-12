#ifndef CONSTS_H
#define CONSTS_H

#include <stdint.h>

#define CONST_WORD_LIMIT 7

#define CONST_CHARSET "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
#define CONST_CHARSET_LENGTH (sizeof(CONST_CHARSET) - 1)

#define CONST_WORD_LENGTH_MIN 1
#define CONST_WORD_LENGTH_MAX 8

#define TOTAL_BLOCKS 16384UL
#define TOTAL_THREADS 512UL
#define HASHES_PER_KERNEL 128UL

#define ERROR_CHECK(X) { gpuAssert((X), __FILE__, __LINE__); }

#endif