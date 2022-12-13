#ifndef CONSTS_H
#define CONSTS_H

#include <stdint.h>

struct device_info{
	struct cudaDeviceProp properties; // Device Properties
	int id; // Device ID
	int max_threads; // Device max threads per block
	int max_blocks; // Device max blocks
	int global_mem; // Global memory length
};

typedef unsigned int UINT32;
#define CONST_WORD_LIMIT 7

#define CONST_CHARSET "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
#define CONST_CHARSET_LENGTH (sizeof(CONST_CHARSET) - 1)
#define CONST_WORD_LENGTH_MAX 8

#define TOTAL_BLOCKS 16384UL
#define TOTAL_THREADS 512UL

#define REQUIRED_SHARED_MEMORY 64
#define FUNCTION_PARAM_ALLOC 256

char potential_chars[] = "abcdefg";
char cur_word[CONST_WORD_LIMIT];
char pwd[CONST_WORD_LIMIT];
__device__ char pwd_d[CONST_WORD_LIMIT];
__device__ char potential_chars_d[CONST_CHARSET_LENGTH];

#define ERROR_CHECK(X) { gpuAssert((X), __FILE__, __LINE__); }

#endif