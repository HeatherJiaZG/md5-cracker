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

char g_word[CONST_WORD_LIMIT];
char g_charset[] = "abcdefg";
char g_cracked[CONST_WORD_LIMIT];

__device__ char g_deviceCharset[CONST_CHARSET_LENGTH];
__device__ char g_deviceCracked[CONST_WORD_LIMIT];

struct deviceInfo{
	struct cudaDeviceProp prop; // Device Properties
	int id; // Device ID
	int max_threads; // Device max threads per block
	int max_blocks; // Device max blocks
	int global_memory_len;
};


#define REQUIRED_SHARED_MEMORY 64
#define FUNCTION_PARAM_ALLOC 256
struct deviceInfo device;


__global__ void md5Crack(uint8_t wordLength, char* charsetWord, UINT32 hash01, UINT32 hash02, UINT32 hash03, UINT32 hash04){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  /* Shared variables */
  __shared__ char sharedCharset[CONST_CHARSET_LENGTH];
  for(int i = 0; i < CONST_CHARSET_LENGTH; i++){
    sharedCharset[i] = g_deviceCharset[i];
  }
  
  /* Thread variables */
  uint8_t threadWordLength = wordLength;
  char threadCharsetWord[CONST_WORD_LIMIT];
  for(int i = 0; i < CONST_WORD_LIMIT; i++){
    threadCharsetWord[i] = charsetWord[i];
  }
  
  /* Increment current word by thread index */
  next(&threadWordLength, threadCharsetWord, idx);

  char threadTextWord[CONST_WORD_LIMIT];
  for(int i = 0; i < threadWordLength; i++){
    threadTextWord[i] = sharedCharset[threadCharsetWord[i]];
  }
  
  struct md5_context context;
  md5_init(&context, (unsigned char*)threadTextWord, threadWordLength);
  md5Hash(&context);   

  if(context.threadHash[0] == hash01 && context.threadHash[1] == hash02 && context.threadHash[2] == hash03 && context.threadHash[3] == hash04){
    for(int i = 0; i < threadWordLength; i++){
      g_deviceCracked[i] = threadTextWord[i];
    }
    return;
  }
  
}


bool runMD5CUDA(char* words, uint8_t g_wordLength, UINT32* hashBins, bool *result, int *time) {
  // true: found, false: not found
  bool found = false;

  // Start Execution Time
  cudaEvent_t start, stop;
  float elapsedTime;
  cudaEventCreate(&start); 
  cudaEventRecord(start, 0);

  /* Copy current data */
  ERROR_CHECK(cudaMemcpy(words, g_word, sizeof(uint8_t) * CONST_WORD_LIMIT, cudaMemcpyHostToDevice)); 
  /* Start kernel */
  md5Crack<<<device.max_blocks, device.max_threads>>>(g_wordLength, words, hashBins[0], hashBins[1], hashBins[2], hashBins[3]);
  /* Global increment */
  *result = next(&g_wordLength, g_word, device.max_threads * device.max_blocks);
    
  /* Synchronize now */
  cudaDeviceSynchronize();
  /* Copy result */
  ERROR_CHECK(cudaMemcpyFromSymbol(g_cracked, g_deviceCracked, sizeof(uint8_t) * CONST_WORD_LIMIT, 0, cudaMemcpyDeviceToHost)); 
 
  /* Check result */
  if(*g_cracked != 0){     
    std::cout << "Notice: cracked " << g_cracked << std::endl; 
    found = true;
  }

  // Stop Execution Time
  cudaEventCreate(&stop);
  cudaEventRecord(stop, 0); 
  cudaEventSynchronize(stop); 
  cudaEventElapsedTime( &elapsedTime, start, stop);
  std::cout << "[DEBUG] elapsedTime = " << elapsedTime << std::endl; 
  *time += elapsedTime;

  return found;
} 

void getOptimalThreads(struct deviceInfo * device) {
	int max_threads = device->prop.maxThreadsPerBlock;
	int max_blocks = 40;
	int shared_memory = device->prop.sharedMemPerBlock - FUNCTION_PARAM_ALLOC;
	
	// calculate the most threads that we can support optimally
	
	while ((shared_memory / max_threads) < REQUIRED_SHARED_MEMORY) { max_threads--; } 	

	device->max_threads = max_threads;		// most threads we support
	device->max_blocks = max_blocks;		// most blocks we support

	// now we need to have (device.max_threads * device.max_blocks) number of words in memory for the graphics card
	
	device->global_memory_len = (device->max_threads * device->max_blocks) * 64;
}


int main(int argc, char* argv[]){

  // FILE *f = fopen("chars.txt", "rb");
  // fseek(f, 0, SEEK_END);
  // long fsize = ftell(f);
  // fseek(f, 0, SEEK_SET);
  
  // char *g_charset = (char *)malloc(fsize + 1);
  // fread(g_charset, fsize, 1, f);
  // fclose(f);
  // g_charset[fsize] = 0;

  int totalTime = 0; 

  device.id = 0;
	cudaGetDeviceProperties(&device.prop, device.id);
  getOptimalThreads(&device);

  /* Hash stored as u32 integers */
  UINT32 hashBins[4];
  getHashBins(argv[1], hashBins);
  
  
  /* Fill memory */
  for (int i=0; i<CONST_WORD_LIMIT; i++) {
    g_word[i] = 0;
    g_cracked[i] = 0;
  }
  
  /* Current word length = minimum word length */
  uint8_t g_wordLength = 1;
  
  /* Current word is different on each device */
  char* words;
    
  /* Copy to each device */
  ERROR_CHECK(cudaMemcpyToSymbol(g_deviceCharset, g_charset, sizeof(uint8_t) * CONST_CHARSET_LENGTH, 0, cudaMemcpyHostToDevice));
  ERROR_CHECK(cudaMemcpyToSymbol(g_deviceCracked, g_cracked, sizeof(uint8_t) * CONST_WORD_LIMIT, 0, cudaMemcpyHostToDevice));
  
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