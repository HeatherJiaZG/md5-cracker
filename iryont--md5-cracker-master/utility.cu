#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <device_functions.h>

#include <stdint.h>
#include <iostream>

#include "consts.h"

void get_hash_bins(char* target, UINT32* hashes) {
  for(int i = 0; i < 4; i++){
    char arr[16];
    
    strncpy(arr, target+i*8, 8);
    sscanf(arr, "%x", &hashes[i]);  
    UINT32 hash1 = (hashes[i] & 0xFF000000);
    UINT32 hash2 = (hashes[i] & 0x00FF0000);
    UINT32 hash3 = (hashes[i] & 0x0000FF00);
    UINT32 hash4 = (hashes[i] & 0x000000FF);
    hash1 = hash1 >> 24;
    hash2 = hash2 >> 8;
    hash3 = hash3 << 8;
    hash4 = hash4 << 24;
    hashes[i] = hash1 | hash2 | hash3 | hash4;
  }
}

__device__ __host__ bool advance_step(uint8_t* len, char* word, UINT32 advance){
  int i = 0;
  UINT32 plus = 0;
  for (i = 0; i<MAX_PWD_LENGTH; i ++) {
    if (advance <= 0) {
      break;
    }
    if(i >= *len && advance > 0){
      advance--;
    }
    plus = advance + word[i];
    word[i] = plus % CHARS_LEN;
    advance = plus / CHARS_LEN;
  }

  if (i > *len){
    *len = i;
  }
  if (i>CONST_WORD_LENGTH_MAX){
    return false;
  }
  return true;
}

