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

void getHashBins(char* target, uint32_t* hashes) {
  for(uint8_t i = 0; i < 4; i++){
    char tmp[16];
    
    strncpy(tmp, target + i * 8, 8);
    sscanf(tmp, "%x", &hashes[i]);  
    uint32_t hash1 = (hashes[i] & 0xFF000000) >> 24;
    uint32_t hash2 = (hashes[i] & 0x00FF0000) >> 8;
    uint32_t hash3 = (hashes[i] & 0x0000FF00) << 8;
    uint32_t hash4 = (hashes[i] & 0x000000FF) << 24;
    hashes[i] = hash1 | hash2 | hash3 | hash4;
  }
}

__device__ __host__ bool next(uint8_t* length, char* word, uint32_t increment){
  uint32_t idx = 0;
  uint32_t add = 0;
  
  while(increment > 0 && idx < CONST_WORD_LIMIT){
    if(idx >= *length && increment > 0){
      increment--;
    }
    
    add = increment + word[idx];
    word[idx] = add % CONST_CHARSET_LENGTH;
    increment = add / CONST_CHARSET_LENGTH;
    idx++;
  }
  
  if(idx > *length){
    *length = idx;
  }
  
  if(idx > CONST_WORD_LENGTH_MAX){
    return false;
  }

  return true;
}

