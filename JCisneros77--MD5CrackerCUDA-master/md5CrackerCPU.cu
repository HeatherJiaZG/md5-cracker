#include <stdio.h>
#include <time.h>
#include "md5LibCPU.cu"

#define WORD_SIZE 512
#define MD5_SIZE 32
#define UINT4 uint

int getNumberOfWords(FILE*);

int main(int argc, char ** argv){
	// Random Seed
	srand(time(NULL));
	// Get file name where MD5 to be cracked is stored
	FILE * inputFile = fopen(argv[1],"r");
	// Get file name for wordlist
	FILE * wordListFile = fopen(argv[2],"r");

	// Get MD5 to be cracked
	char md5ToCrack[MD5_SIZE+1];
	fgets(md5ToCrack,MD5_SIZE+1,inputFile);

	// Close input File
	fclose(inputFile);

	// Initialize word list size
	int wordListSize = getNumberOfWords(wordListFile);
  // Reset the file pointer to word list
	rewind(wordListFile);

	// Word found index
	int wordFound = -1;
	//Start Total Time
	  cudaEvent_t startTotal, stopTotal;
	  float elapsedTimeTotal;
	  cudaEventCreate(&startTotal);
	  cudaEventRecord(startTotal,0);
 		
    // Start Execution Time
 	  cudaEvent_t start, stop;
 		float elapsedTime;
 		cudaEventCreate(&start); 
		cudaEventRecord(start, 0);
		char currentWord[WORD_SIZE+1];
    // Counter to see the number of iterations
		int cont = 0;
    // Repeat for every word in wordlist or until the right word is found
		for (int i = 0; i < wordListSize; ++i){
			// Var where MD5 will be stored
      char hash[33];
			cont++;
      // Get word from wordlist File
			fgets(currentWord,WORD_SIZE+1,wordListFile);
          // Remove spaces or end of line chars from word
          for (char *s = &(currentWord[0]); s < &(currentWord[WORD_SIZE+1]); ++s) {
             if ('\r' == *s || '\n' == *s || '\0' == *s) {
                *s = '\0'; break;
             }
          }
			//Get MD5 for currentword and store in hash
			MDString(currentWord,hash);
      // Check if MD5's are equal
			if (strcmp(hash,md5ToCrack) == 0){
        // MD5 cracked
				// Change flag to True and break cycle
        wordFound = 1;
				break; 
			}
		}
 		// Stop Execution Time
 		cudaEventCreate(&stop);
 		cudaEventRecord(stop, 0); 
 		cudaEventSynchronize(stop); 
 		cudaEventElapsedTime( &elapsedTime, start, stop);
 		printf("Time to crack MD5 : %f ms\n" ,elapsedTime);
 		
    // Check if word was found and print
 		if (wordFound == -1)
 			printf("Wasn't able to crack MD5\n");
 		else
 			printf("-------------MD5 Cracked!-------------\nWord: %s\n--------------------------------------\n",currentWord);

	// Close word list file
	fclose(wordListFile);
   	// Stop Total time
	cudaEventCreate(&stopTotal);
	cudaEventRecord(stopTotal,0);
	cudaEventSynchronize(stopTotal);
	cudaEventElapsedTime(&elapsedTimeTotal, startTotal,stopTotal);
	printf("Total Time : %f ms\n" ,elapsedTimeTotal);
	return 0;
}

// Return number of words from wordlist File
int getNumberOfWords(FILE * fp){
	int lines = 1;
	int ch = 0;
	while(!feof(fp))
	{
	  ch = fgetc(fp);
	  if(ch == '\n')
	  {
	    lines++;
	  }
	}
	return lines;
}



