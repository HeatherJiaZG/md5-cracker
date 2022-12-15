#include <stdlib.h>
#include <stdint.h>
#include <sstream>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <string.h>
#include <curand_kernel.h>
#include <device_functions.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "consts.h"
#include "utility.cu"
#include "lib_md5.cu"
#include "lib_md5.h"


__global__ void md5_cuda(uint8_t pwd_len, char* words, UINT32* hashBins){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  bool hash_match = true;
  
  // init shared mem
  __shared__ char shared_mem[CHARS_LEN];
  for(int i = 0; i < CHARS_LEN; i++){
    shared_mem[i] = potential_chars_d[i];
  }
  
  // for threads
  uint8_t word_len = pwd_len;
  char cur_word[MAX_PWD_LENGTH];
  for(int i = 0; i < MAX_PWD_LENGTH; i++){
    cur_word[i] = words[i];
  }
  
  // increment the word to crack
  advance_step(&word_len, cur_word, idx);

  // get word for thread
  char thread_word[MAX_PWD_LENGTH];
  for(int i = 0; i < word_len; i++){
    thread_word[i] = shared_mem[cur_word[i]];
  }
  
  // call md5 functions
  struct md5_states context;
  md5_init(&context, (unsigned char*)thread_word, word_len);
  md5_run(&context);   

  for(int i = 0; i < word_len; i++){
    bool current_match = context.hashes[i] == hashBins[i];
    hash_match = hash_match && current_match;
  }
  // get result hash
  if(hash_match){
    for(int i = 0; i < word_len; i++){
      pwd_d[i] = thread_word[i];
    }
    return;
  }
  
}