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

