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

  // copy data to device and call cuda md5 function
  ERROR_CHECK(cudaMemcpy(words, cur_word, sizeof(uint8_t) * MAX_PWD_LENGTH, cudaMemcpyHostToDevice)); 
  md5_cuda<<<device.max_blocks, device.max_threads>>>(pwd_len, words, hashBins_d);
  *result = advance_step(&pwd_len, cur_word, device.max_threads * device.max_blocks);
  cudaDeviceSynchronize();
  ERROR_CHECK(cudaMemcpyFromSymbol(pwd, pwd_d, sizeof(uint8_t) * MAX_PWD_LENGTH, 0, cudaMemcpyDeviceToHost)); 
 
  // password cracked
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

	// have (device.max_threads * device.max_blocks) number of words in memory for the GPU
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

  UINT32 hash_bins[4];
  get_hash_bins(argv[1], hash_bins);
  
  for (int i=0; i<MAX_PWD_LENGTH; i++) {
    cur_word[i] = 0;
    pwd[i] = 0;
  }
  
  char* words;
  ERROR_CHECK(cudaMalloc((void**)&words, sizeof(uint8_t) * MAX_PWD_LENGTH));
  ERROR_CHECK(cudaMemcpyToSymbol(potential_chars_d, potential_chars, sizeof(uint8_t) * CHARS_LEN, 0, cudaMemcpyHostToDevice));
  ERROR_CHECK(cudaMemcpyToSymbol(pwd_d, pwd, sizeof(uint8_t) * MAX_PWD_LENGTH, 0, cudaMemcpyHostToDevice));

  while(result && !found){
    found = runMD5CUDA(words, pwd_len, hash_bins, &result, &total_time);
  }

  if(!result && !found){
    std::cout << "No matched password.\n";
  }

  cudaFree(words);  
  std::cout << "Time spent: " << total_time << " ms\n";
  
}