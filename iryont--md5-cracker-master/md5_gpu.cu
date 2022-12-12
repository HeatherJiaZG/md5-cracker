/**
 * CUDA MD5 cracker
 * Copyright (C) 2015  Konrad Kusnierz <iryont@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <stdio.h>
#include <iostream>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <sstream>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <device_functions.h>

#include "consts.h"
#include "utility.cu"
#include "md5.cu"

char g_word[CONST_WORD_LIMIT];
char g_charset[CONST_CHARSET_LIMIT];
char g_cracked[CONST_WORD_LIMIT];

__device__ char g_deviceCharset[CONST_CHARSET_LIMIT];
__device__ char g_deviceCracked[CONST_WORD_LIMIT];

__global__ void md5Crack(uint8_t wordLength, char* charsetWord, uint32_t hash01, uint32_t hash02, uint32_t hash03, uint32_t hash04){
  uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * HASHES_PER_KERNEL;
  
  /* Shared variables */
  __shared__ char sharedCharset[CONST_CHARSET_LIMIT];
  
  /* Thread variables */
  char threadCharsetWord[CONST_WORD_LIMIT];
  char threadTextWord[CONST_WORD_LIMIT];
  uint8_t threadWordLength;
  uint32_t threadHash01, threadHash02, threadHash03, threadHash04;
  
  /* Copy everything to local memory */
  memcpy(threadCharsetWord, charsetWord, CONST_WORD_LIMIT);
  memcpy(&threadWordLength, &wordLength, sizeof(uint8_t));
  memcpy(sharedCharset, g_deviceCharset, sizeof(uint8_t) * CONST_CHARSET_LIMIT);
  
  /* Increment current word by thread index */
  next(&threadWordLength, threadCharsetWord, idx);
  
  for(uint32_t hash = 0; hash < HASHES_PER_KERNEL; hash++){
    for(uint32_t i = 0; i < threadWordLength; i++){
      threadTextWord[i] = sharedCharset[threadCharsetWord[i]];
    }
    
    md5Hash((unsigned char*)threadTextWord, threadWordLength, &threadHash01, &threadHash02, &threadHash03, &threadHash04);   

    if(threadHash01 == hash01 && threadHash02 == hash02 && threadHash03 == hash03 && threadHash04 == hash04){
      memcpy(g_deviceCracked, threadTextWord, threadWordLength);
    }
    
    if(!next(&threadWordLength, threadCharsetWord, 1)){
      break;
    }
  }
}

int main(int argc, char* argv[]){
  /* Check arguments */
  if(argc != 2 || strlen(argv[1]) != 32){
    std::cout << argv[0] << " <md5_hash>" << std::endl;
    return -1;
  }
  
  /* Hash stored as u32 integers */
  uint32_t md5Hash[4];
  
  /* Parse argument */
  for(uint8_t i = 0; i < 4; i++){
    char tmp[16];
    
    strncpy(tmp, argv[1] + i * 8, 8);
    sscanf(tmp, "%x", &md5Hash[i]);   
    md5Hash[i] = (md5Hash[i] & 0xFF000000) >> 24 | (md5Hash[i] & 0x00FF0000) >> 8 | (md5Hash[i] & 0x0000FF00) << 8 | (md5Hash[i] & 0x000000FF) << 24;
  }
  
  /* Fill memory */
  memset(g_word, 0, CONST_WORD_LIMIT);
  memset(g_cracked, 0, CONST_WORD_LIMIT);
  memcpy(g_charset, CONST_CHARSET, CONST_CHARSET_LENGTH);
  
  /* Current word length = minimum word length */
  uint8_t g_wordLength = CONST_WORD_LENGTH_MIN;
  
  /* Time */
  cudaEvent_t clockBegin;
  cudaEvent_t clockLast;
  
  cudaEventCreate(&clockBegin);
  cudaEventCreate(&clockLast);
  cudaEventRecord(clockBegin, 0);
  
  /* Current word is different on each device */
  char** words = new char*[1];

    
  /* Copy to each device */
  ERROR_CHECK(cudaMemcpyToSymbol(g_deviceCharset, g_charset, sizeof(uint8_t) * CONST_CHARSET_LIMIT, 0, cudaMemcpyHostToDevice));
  ERROR_CHECK(cudaMemcpyToSymbol(g_deviceCracked, g_cracked, sizeof(uint8_t) * CONST_WORD_LIMIT, 0, cudaMemcpyHostToDevice));
  
  /* Allocate on each device */
  ERROR_CHECK(cudaMalloc((void**)&words[0], sizeof(uint8_t) * CONST_WORD_LIMIT));

  bool result, found;
  while(!found){
    result = false;
    found = false;
    
    /* Copy current data */
    ERROR_CHECK(cudaMemcpy(words[0], g_word, sizeof(uint8_t) * CONST_WORD_LIMIT, cudaMemcpyHostToDevice)); 
    /* Start kernel */
    md5Crack<<<TOTAL_BLOCKS, TOTAL_THREADS>>>(g_wordLength, words[0], md5Hash[0], md5Hash[1], md5Hash[2], md5Hash[3]);
    /* Global increment */
    result = next(&g_wordLength, g_word, TOTAL_THREADS * HASHES_PER_KERNEL * TOTAL_BLOCKS);
    
    /* Display progress */
    char word[CONST_WORD_LIMIT];
    
    for(int i = 0; i < g_wordLength; i++){
      word[i] = g_charset[g_word[i]];
    }
    
    std::cout << "Notice: currently at " << std::string(word, g_wordLength) << " (" << (uint32_t)g_wordLength << ")" << std::endl;
      
    /* Synchronize now */
    cudaDeviceSynchronize();
    /* Copy result */
    ERROR_CHECK(cudaMemcpyFromSymbol(g_cracked, g_deviceCracked, sizeof(uint8_t) * CONST_WORD_LIMIT, 0, cudaMemcpyDeviceToHost)); 
    /* Check result */
    found = *g_cracked;
    // if(found = *g_cracked != 0){     
    //   std::cout << "Notice: cracked " << g_cracked << std::endl; 
    //   break;
    // }
    
    if(!result){
      break;
    }
  }

  if(!result && !found) {
    std::cout << "Notice: found nothing (host)" << std::endl;
  } else {
    std::cout << "Notice: cracked " << g_cracked << std::endl; 
  }
    
  /* Free on each device */
  cudaFree((void**)words[0]);
  
  /* Free array */
  delete[] words;
  
  float milliseconds = 0;
  
  cudaEventRecord(clockLast, 0);
  cudaEventSynchronize(clockLast);
  cudaEventElapsedTime(&milliseconds, clockBegin, clockLast);
  
  std::cout << "Notice: computation time " << milliseconds << " ms" << std::endl;
  
  cudaEventDestroy(clockBegin);
  cudaEventDestroy(clockLast);
}
