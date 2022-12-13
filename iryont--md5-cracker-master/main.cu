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
#include "lib_md5.cu"
#include "lib_md5.h"

// char potential_chars[] = "abcdefg";
// char cur_word[CONST_WORD_LIMIT];
// char pwd[CONST_WORD_LIMIT];
// __device__ char pwd_d[CONST_WORD_LIMIT];
// __device__ char potential_chars_d[CONST_CHARSET_LENGTH];


__global__ void md5_cuda(uint8_t wordLength, char* charsetWord, UINT32 hash01, UINT32 hash02, UINT32 hash03, UINT32 hash04){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  /* Shared variables */
  __shared__ char sharedCharset[CONST_CHARSET_LENGTH];
  for(int i = 0; i < CONST_CHARSET_LENGTH; i++){
    sharedCharset[i] = potential_chars_d[i];
  }
  
  /* Thread variables */
  uint8_t threadWordLength = wordLength;
  char threadCharsetWord[CONST_WORD_LIMIT];
  for(int i = 0; i < CONST_WORD_LIMIT; i++){
    threadCharsetWord[i] = charsetWord[i];
  }
  
  /* Increment current word by thread index */
  advance_step(&threadWordLength, threadCharsetWord, idx);

  char threadTextWord[CONST_WORD_LIMIT];
  for(int i = 0; i < threadWordLength; i++){
    threadTextWord[i] = sharedCharset[threadCharsetWord[i]];
  }
  
  struct md5_states context;
  md5_init(&context, (unsigned char*)threadTextWord, threadWordLength);
  md5_run(&context);   

  if(context.hashes[0] == hash01 && context.hashes[1] == hash02 && context.hashes[2] == hash03 && context.hashes[3] == hash04){
    for(int i = 0; i < threadWordLength; i++){
      pwd_d[i] = threadTextWord[i];
    }
    return;
  }
  
}

struct device_info device;

bool runMD5CUDA(char* words, uint8_t g_wordLength, UINT32* hashBins, bool *result, float *time) {
  // true: found, false: not found
  bool found = false;

  // Start Execution Time
  cudaEvent_t start, stop;
  float elapsedTime;
  cudaEventCreate(&start); 
  cudaEventRecord(start, 0);

  /* Copy current data */
  ERROR_CHECK(cudaMemcpy(words, cur_word, sizeof(uint8_t) * CONST_WORD_LIMIT, cudaMemcpyHostToDevice)); 
  /* Start kernel */
  md5_cuda<<<device.max_blocks, device.max_threads>>>(g_wordLength, words, hashBins[0], hashBins[1], hashBins[2], hashBins[3]);
  /* Global increment */
  *result = advance_step(&g_wordLength, cur_word, device.max_threads * device.max_blocks);
    
  /* Synchronize now */
  cudaDeviceSynchronize();
  /* Copy result */
  ERROR_CHECK(cudaMemcpyFromSymbol(pwd, pwd_d, sizeof(uint8_t) * CONST_WORD_LIMIT, 0, cudaMemcpyDeviceToHost)); 
 
  /* Check result */
  if(*pwd != 0){     
    std::cout << "Notice: cracked " << pwd << std::endl; 
    found = true;
  }

  // Stop Execution Time
  cudaEventCreate(&stop);
  cudaEventRecord(stop, 0); 
  cudaEventSynchronize(stop); 
  cudaEventElapsedTime( &elapsedTime, start, stop); 
  *time += elapsedTime;

  return found;
} 

void getOptimalThreads(struct device_info * device) {
	int max_threads = device->properties.maxThreadsPerBlock;
	int max_blocks = 40;
	int shared_memory = device->properties.sharedMemPerBlock - FUNCTION_PARAM_ALLOC;
	
	// calculate the most threads that we can support optimally
	
	while ((shared_memory / max_threads) < REQUIRED_SHARED_MEMORY) { max_threads--; } 	

	device->max_threads = max_threads;		// most threads we support
	device->max_blocks = max_blocks;		// most blocks we support

	// now we need to have (device.max_threads * device.max_blocks) number of words in memory for the graphics card
	
	device->global_mem = (device->max_threads * device->max_blocks) * 64;
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

  float totalTime = 0; 

  device.id = 0;
	cudaGetDeviceProperties(&device.properties, device.id);
  getOptimalThreads(&device);

  /* Hash stored as u32 integers */
  UINT32 hashBins[4];
  get_hash_bins(argv[1], hashBins);
  
  
  /* Fill memory */
  for (int i=0; i<CONST_WORD_LIMIT; i++) {
    cur_word[i] = 0;
    pwd[i] = 0;
  }
  
  /* Current word length = minimum word length */
  uint8_t g_wordLength = 1;
  
  /* Current word is different on each device */
  char* words;
    
  /* Copy to each device */
  ERROR_CHECK(cudaMemcpyToSymbol(potential_chars_d, potential_chars, sizeof(uint8_t) * CONST_CHARSET_LENGTH, 0, cudaMemcpyHostToDevice));
  ERROR_CHECK(cudaMemcpyToSymbol(pwd_d, pwd, sizeof(uint8_t) * CONST_WORD_LIMIT, 0, cudaMemcpyHostToDevice));
  
  /* Allocate on each device */
  ERROR_CHECK(cudaMalloc((void**)&words, sizeof(uint8_t) * CONST_WORD_LIMIT));

  bool result = true;
  bool found = false;

  while(result && !found){
    found = runMD5CUDA(words, g_wordLength, hashBins, &result, &totalTime);
  }

  if(!result && !found){
    std::cout << "Notice: found nothing (host)" << std::endl;
  }
    
  /* Free on each device */
  cudaFree(words);
  
  std::cout << "Notice: computation time " << totalTime << " ms" << std::endl;
  
}