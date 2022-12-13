#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <device_functions.h>

#include <stdint.h>
#include <iostream>

#include "consts.h"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true){
  if(code != cudaSuccess){
    std::cout << "Error: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
    if(abort){
      exit(code);
    }
  }
}

void getHashBins(char* target, UINT32* hashes) {
  for(int i = 0; i < 4; i++){
    char tmp[16];
    
    strncpy(tmp, target + i * 8, 8);
    sscanf(tmp, "%x", &hashes[i]);  
    UINT32 hash1 = (hashes[i] & 0xFF000000);
    hash1 = hash1 >> 24;
    UINT32 hash2 = (hashes[i] & 0x00FF0000);
    hash2 = hash2 >> 8;
    UINT32 hash3 = (hashes[i] & 0x0000FF00);
    hash3 = hash3 << 8;
    UINT32 hash4 = (hashes[i] & 0x000000FF);
    hash4 = hash4 << 24;
    hashes[i] = hash1 | hash2 | hash3 | hash4;
  }
}

__device__ __host__ bool advance_step(uint8_t* length, char* word, UINT32 increment){
  int i = 0;
  UINT32 add = 0;
  for (i = 0; i<CONST_WORD_LIMIT; i ++) {
    if (increment <= 0) {
      break;
    }
    if(i >= *length && increment > 0){
      increment--;
    }
    add = increment + word[i];
    word[i] = add % CONST_CHARSET_LENGTH;
    increment = add / CONST_CHARSET_LENGTH;
  }

  if(i > *length){
    *length = i;
  }
  if(i > CONST_WORD_LENGTH_MAX){
    return false;
  }
  return true;
}

