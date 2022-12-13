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
  
  /* Shared variables */
  __shared__ char shared_mem[CONST_CHARSET_LENGTH];
  for(int i = 0; i < CONST_CHARSET_LENGTH; i++){
    shared_mem[i] = potential_chars_d[i];
  }
  
  /* Thread variables */
  uint8_t word_len = pwd_len;
  char cur_word[CONST_WORD_LIMIT];
  for(int i = 0; i < CONST_WORD_LIMIT; i++){
    cur_word[i] = words[i];
  }
  
  /* Increment current word by thread index */
  advance_step(&word_len, cur_word, idx);

  char thread_word[CONST_WORD_LIMIT];
  for(int i = 0; i < word_len; i++){
    thread_word[i] = shared_mem[cur_word[i]];
  }
  
  struct md5_states context;
  md5_init(&context, (unsigned char*)thread_word, word_len);
  md5_run(&context);   

  for(int i = 0; i < word_len; i++){
    bool current_match = context.hashes[i] == hashBins[i];
    hash_match = hash_match && current_match;
  }
  if(hash_match){
    for(int i = 0; i < word_len; i++){
      pwd_d[i] = thread_word[i];
    }
    return;
  }
  
}

struct device_info device;

bool runMD5CUDA(char* words, uint8_t pwd_len, UINT32* hashBins, bool *result, float *time) {
  // true: found, false: not found
  bool found = false;
  UINT32 *hashBins_d;
  cudaMalloc((void **)&hashBins_d, sizeof(UINT32) * 4);
  cudaMemcpy(hashBins_d, hashBins, sizeof(UINT32) * 4, cudaMemcpyHostToDevice);

  // Start Execution Time
  cudaEvent_t start, stop;
  float elapsedTime;
  cudaEventCreate(&start); 
  cudaEventRecord(start, 0);

  /* Copy current data */
  ERROR_CHECK(cudaMemcpy(words, cur_word, sizeof(uint8_t) * CONST_WORD_LIMIT, cudaMemcpyHostToDevice)); 
  /* Start kernel */
  md5_cuda<<<device.max_blocks, device.max_threads>>>(pwd_len, words, hashBins_d);
  /* Global increment */
  *result = advance_step(&pwd_len, cur_word, device.max_threads * device.max_blocks);
    
  /* Synchronize now */
  cudaDeviceSynchronize();
  /* Copy result */
  ERROR_CHECK(cudaMemcpyFromSymbol(pwd, pwd_d, sizeof(uint8_t) * CONST_WORD_LIMIT, 0, cudaMemcpyDeviceToHost)); 
 
  /* Check result */
  if(*pwd != 0){    
    found = true; 
    std::cout << "The cracked word is: " << pwd << std::endl; 
  }

  // Stop Execution Time
  cudaEventCreate(&stop);
  cudaEventRecord(stop, 0); 
  cudaEventSynchronize(stop); 
  cudaEventElapsedTime( &elapsedTime, start, stop); 
  *time += elapsedTime;

  return found;
} 

void get_optimal_threads(struct device_info * device) {
  int max_blocks = 40;
	int max_threads = device->properties.maxThreadsPerBlock;
	int shared_memory = device->properties.sharedMemPerBlock - FUNCTION_PARAM_ALLOC;
	
	// calculate the most threads that we can support optimally
	
	while ((shared_memory / max_threads) < REQUIRED_SHARED_MEMORY) {
    max_threads--; 
  } 	

	device->max_threads = max_threads;
	device->max_blocks = max_blocks;
  device->global_mem = (device->max_threads * device->max_blocks) * 64;

	// now we need to have (device.max_threads * device.max_blocks) number of words in memory for the GPU
}


int main(int argc, char* argv[]){

  // FILE *f = fopen("chars.txt", "rb");
  // fseek(f, 0, SEEK_END);
  // long fsize = ftell(f);
  // fseek(f, 0, SEEK_SET);
  
  // char *potential_chars = (char *)malloc(fsize + 1);
  // fread(potential_chars, fsize, 1, f);
  // fclose(f);
  // potential_chars[fsize] = 0;

  float total_time = 0; 
  uint8_t pwd_len = 1;
  bool result = true;
  bool found = false;

  device.id = 0;
	cudaGetDeviceProperties(&device.properties, device.id);
  get_optimal_threads(&device);

  /* Hash stored as u32 integers */
  UINT32 hash_bins[4];
  get_hash_bins(argv[1], hash_bins);
  
  
  /* Fill memory */
  for (int i=0; i<CONST_WORD_LIMIT; i++) {
    cur_word[i] = 0;
    pwd[i] = 0;
  }
  
  /* Current word length = minimum word length */
  
  /* Current word is different on each device */
  char* words;
  ERROR_CHECK(cudaMalloc((void**)&words, sizeof(uint8_t) * CONST_WORD_LIMIT));
    
  /* Copy to each device */
  ERROR_CHECK(cudaMemcpyToSymbol(potential_chars_d, potential_chars, sizeof(uint8_t) * CONST_CHARSET_LENGTH, 0, cudaMemcpyHostToDevice));
  ERROR_CHECK(cudaMemcpyToSymbol(pwd_d, pwd, sizeof(uint8_t) * CONST_WORD_LIMIT, 0, cudaMemcpyHostToDevice));

  while(result && !found){
    found = runMD5CUDA(words, pwd_len, hash_bins, &result, &total_time);
  }

  if(!result && !found){
    std::cout << "No matched password.\n";
  }

  cudaFree(words);  
  std::cout << "Time: " << total_time << " ms\n";
  
}