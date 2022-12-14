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

#define MAX_PWD_LENGTH 7
#define CHARS_LEN 6
#define CONST_WORD_LENGTH_MAX 7
#define FUNCTION_PARAM_ALLOC 256

__device__ char pwd_d[MAX_PWD_LENGTH];
__device__ char potential_chars_d[CHARS_LEN];

char potential_chars[] = "abcdefg";
char cur_word[MAX_PWD_LENGTH];
char pwd[MAX_PWD_LENGTH];

#define ERROR_CHECK(X) { gpuAssert((X), __FILE__, __LINE__); }

#endif