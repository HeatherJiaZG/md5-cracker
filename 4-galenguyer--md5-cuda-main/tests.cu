#include <cuda.h>
#include <stdint.h>
#include <string.h>
#include "md5.cu"

#include <string>
#include <iostream>

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


char* md5_hash(const char* h_str) {
    char* d_str;
    unsigned char* h_res = (unsigned char*)malloc(sizeof(unsigned char)*(32 + 1));
    unsigned char* d_res;
    cudaMalloc((void**)&d_str, sizeof(char) * strlen(h_str));
    cudaMalloc((void**)&d_res, sizeof(char) * 32);
    cudaMemcpy(d_str, h_str, sizeof(char) * strlen(h_str), cudaMemcpyHostToDevice);

    md5<<<device.max_blocks, device.max_threads>>>(d_str, (uint32_t)strlen(h_str), d_res);

    cudaMemcpy(h_res, d_res, sizeof(unsigned char)*(32), cudaMemcpyDeviceToHost);

    cudaFree(d_str);
    cudaFree(d_res);

    char* res = (char*)malloc(sizeof(char)*32);
    for (int i = 0; i < 16; i++) {
        sprintf(&res[i*2], "%2.2x", h_res[i]);
    }
    return res;
}

int run_test(const char* name, const char* result, const char* expected) {
    if (strcmp(expected, result) == 0) {
        printf("TEST PASSED: %s: expected %s, got %s\n", name, expected, result);
        return 1;
    } else {
        printf("TEST FAILED: %s: expected %s, got %s\n", name, expected, result);
        return 0;
    }
}

char *target = "07159c47ee1b19ae4fb9c40d480856c4";

void print_str(const char str[],std::string prefix,const int n, const int lenght) {
        if (lenght == 1)
            {
                for (int j = 0; j < n; j++) {
                    if (strcmp(target, md5_hash(prefix + str[j])) == 0) {
                        std::cout << "Cracked = " << prefix + str[j] << std::endl;
                        break
                    }
                }
                // std::cout << prefix + str[j] << std::endl;

            }//Base case: lenght = 1, print the string "lenght" times + the remaining letter

        else
            {
               // One by one add all characters from "str" and recursively call for "lenght" equals to "lenght"-1
                for (int i = 0; i < n; i++)

                // Next character of input added
                print_str(str, prefix + str[i], n, lenght - 1);
                // "lenght" is decreased, because we have added a new character

            }

    }


int main() {

	// Set device id to 0 (Use fastest device)
	device.id = 0;
	cudaGetDeviceProperties(&device.prop, device.id);
    getOptimalThreads(&device);

    int passed = 0, failed = 0;

    // run_test("md5(\"\")", hash(""), "d41d8cd98f00b204e9800998ecf8427e") ? passed++ : failed++;
    // run_test("md5(\"a\")", hash("a"), "0cc175b9c0f1b6a831c399e269772661") ? passed++ : failed++;
    run_test("md5(\"abc\")", md5_hash("abc"), "900150983cd24fb0d6963f7d28e17f72") ? passed++ : failed++;
    run_test("md5(\"ba\")", md5_hash("ba"), "07159c47ee1b19ae4fb9c40d480856c4") ? passed++ : failed++;
    // run_test("md5(\"message digest\")", hash("message digest"), "f96b697d7cb7938d525a2f31aaf161d0") ? passed++ : failed++;
    // run_test("md5(\"abcdefghijklmnopqrstuvwxyz\")", \
    //     hash("abcdefghijklmnopqrstuvwxyz"), \
    //     "c3fcd3d76192e4007dfb496cca67e13b") ? passed++ : failed++;
    // run_test("md5(\"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789\")", \
    //     hash("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"), \
    //     "d174ab98d277d9f5a5611c2c9f419d9f") ? passed++ : failed++;
    // run_test("md5(\"12345678901234567890123456789012345678901234567890123456789012345678901234567890\")", \
    //     hash("12345678901234567890123456789012345678901234567890123456789012345678901234567890"), \
    //     "57edf4a22be3c955ac49da2e2107b67a") ? passed++ : failed++;

    printf("Tests Passed: %i\n", passed);
    printf("Tests Failed: %i\n", failed);



    int len = 2;
    char str[] = {'a', 'b', 'c', 'd'};
    int n = sizeof(str);
    print_str(str, "", n, len);


    return failed;
}
