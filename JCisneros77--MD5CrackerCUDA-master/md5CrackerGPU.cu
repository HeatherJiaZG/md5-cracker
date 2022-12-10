#include <stdio.h>
#include <cuda.h>
#include <string.h>
#include <arpa/inet.h>
#include "md5LibGPU.cu"
#include <ctype.h>

#define WORD_SIZE 64
#define MD5_SIZE 32
// Function to call CUDA function in .cu file
extern int callMD5CUDA(struct deviceInfo *,char *, int *,int *, int,int *);
// Helper Functions
void getOptimalThreads(struct deviceInfo *);

int main(int argc, char ** argv){
	// Initialize total time in 0
	totalTime = 0;
	// Initialize deviceInfo struct
	struct deviceInfo device; 
	// Set device id to 0 (Use fastest device)
	device.id = 0;
	// Get device properties
	cudaGetDeviceProperties(&device.prop, device.id);
	// Get MD5 To be cracked 				*
	FILE * inputFile = fopen(argv[1],"r");//*
	char md5ToCrack[MD5_SIZE+1];//			*
	fgets(md5ToCrack,MD5_SIZE+1,inputFile);//*
	fclose(inputFile); // Close input file

	// Turn input MD5 into hexadecimal array
	int md5Int[16];
	memset(md5Int, 0, sizeof(md5Int));
	for (int i = 0; i < 16; ++i){
		char * string = (char *) malloc(2*sizeof(char));
		string[0] = md5ToCrack[2*i];
		string[1] = md5ToCrack[(2*i)+1];
		int test = strtol (string,NULL,16);
		md5Int[i] = test;
		free(string);
	}
	// Get file name for wordlist and open
	FILE * wordListFile = fopen(argv[2],"r");
	// Get optimal threads for selected device
	getOptimalThreads(&device);

	// Copy input hash to GPU
	int * target_hash;
	cudaMalloc((void **) &target_hash, sizeof(int) * 16);
	cudaMemcpy(target_hash, md5Int, sizeof(int) * 16, cudaMemcpyHostToDevice);

	// Read file and call cuda as many times as necessary
	char currentWord[WORD_SIZE+1];
	int flagCycle = 1;
  	int hash_found = 0;
	while(flagCycle == 1 && hash_found == 0){
		// Initialize words array, lengths and number of words 
		char ** words = (char**) malloc(sizeof(char*));
		int * wordLengths = (int *) malloc( sizeof(int));
		// Current buffer in 0
		int currentBuffer = 0;
		int numberOfWords = 0;
		// Get set of words for this iteration as long as 
		// there's still space in GPU for more words
		while(numberOfWords < (device.max_threads*device.max_blocks)){
			if (fgets(currentWord,WORD_SIZE+1,wordListFile) != NULL){
				// Remove spaces or end of line chars from word
	            for (char *s = &(currentWord[0]); s < &(currentWord[WORD_SIZE+1]); ++s) {
	               if ('\r' == *s || '\n' == *s || '\0' == *s) {
	                  *s = '\0'; break;
	               }
	            }
            	// Reallocate words array to add new word
				numberOfWords++;
				wordLengths = (int *) realloc(wordLengths,(numberOfWords * sizeof(int)));
				words = (char**)realloc(words, (numberOfWords*sizeof(char*)));
			    wordLengths[numberOfWords-1] = strlen(currentWord);
			    words[numberOfWords-1] = (char*)malloc(sizeof(currentWord));
			    strcpy(words[numberOfWords-1], currentWord);	
			    currentBuffer+= strlen(currentWord);			
			}
			else{ // No more words in file
				// MD5 was not found.
				if (numberOfWords == 0){
					// Set flag to 0 to stop looking in file
					printf("Wasn't able to crack MD5\n");
					flagCycle = 0;
				}
				break;
			}
		}
		if (flagCycle == 1){
			// Initialize arrays to use for GPU 
			// Turn char** to a large char*
			char * flatWords = (char *) malloc(currentBuffer * sizeof(char));
			unsigned * ind = (unsigned *)malloc(numberOfWords * sizeof(unsigned));
			unsigned next = 0;
			// Get all the words into a single array
			// Saving each of their start indexes
			for (int i = 0; i < numberOfWords; ++i){
				strcpy(flatWords+next,words[i]);
				ind[i] = next;
				next += wordLengths[i];
				free(words[i]);
			}
			// Free memory
			free(words);
			// Initialize arrays on device
			char * d_flatWords;
			int * d_ind;
			int * d_wordLengths;
			// Allocate memory
			cudaMalloc((void**)&d_flatWords, next * sizeof(char));
			cudaMalloc((void**)&d_ind,numberOfWords * sizeof(int));
			cudaMalloc((void**)&d_wordLengths,numberOfWords * sizeof(int));
			// Copy memory from CPU to GPU
			cudaMemcpy(d_flatWords,flatWords,next * sizeof(char),cudaMemcpyHostToDevice);
			cudaMemcpy(d_ind, ind, numberOfWords * sizeof(int),cudaMemcpyHostToDevice);
			cudaMemcpy(d_wordLengths, wordLengths, numberOfWords * sizeof(int),cudaMemcpyHostToDevice);
			// Call GPU kernel
			hash_found = callMD5CUDA(&device,d_flatWords,target_hash,d_ind,numberOfWords,d_wordLengths);
			// Free memory
			cudaFree(d_flatWords);
	   		cudaFree(d_ind);
	   		cudaFree(d_wordLengths);
	   		free(flatWords);
	   		free(ind);
   			free(wordLengths);
   		}else{
   			// Free memory
   			free(words);
   			free(wordLengths);
   		}

		// Check if word was found and print 
		if (hash_found == 1) {
			// If word was found, stop cycle 
			flagCycle = 0;
		}
	}
	// Print total time
	printf("Total Time: %f ms\n",totalTime);
	// Close word list file
	fclose(wordListFile);
	return 0;

}
/************** Optimize Cuda Threads Helper **************/
#define REQUIRED_SHARED_MEMORY 64
#define FUNCTION_PARAM_ALLOC 256

void getOptimalThreads(struct deviceInfo * device) {
	int max_threads;
	int max_blocks;
	int shared_memory;

	max_threads = device->prop.maxThreadsPerBlock;
	shared_memory = device->prop.sharedMemPerBlock - FUNCTION_PARAM_ALLOC;
	
	// calculate the most threads that we can support optimally
	
	while ((shared_memory / max_threads) < REQUIRED_SHARED_MEMORY) { max_threads--; } 

	// now we spread our threads across blocks 
	
	max_blocks = 40;		

	device->max_threads = max_threads;		// most threads we support
	device->max_blocks = max_blocks;		// most blocks we support

	// now we need to have (device.max_threads * device.max_blocks) number of words in memory for the graphics card
	
	device->global_memory_len = (device->max_threads * device->max_blocks) * 64;
}