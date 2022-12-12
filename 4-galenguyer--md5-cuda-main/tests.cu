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


char* md5_hash(const char* h_str, int pwd_len) {
    char* d_str;
    unsigned char* h_res = (unsigned char*)malloc(sizeof(unsigned char)*(32 + 1));
    unsigned char* d_res;
    char* d_res_converted;
    char* cracked_pwd;
    char* h_cracked_pwd = (char*)malloc(sizeof(char)*(pwd_len + 1));
    cudaMalloc((void**)&d_str, sizeof(char) * strlen(h_str));
    cudaMalloc((void**)&d_res, sizeof(char) * 32);
    cudaMalloc((void**)&d_res_converted, sizeof(char) * 32);
    cudaMalloc((void**)&cracked_pwd, sizeof(char) * pwd_len);
    cudaMemcpy(d_str, h_str, sizeof(char) * strlen(h_str), cudaMemcpyHostToDevice);

    md5<<<device.max_blocks, device.max_threads>>>(d_str, (uint32_t)pwd_len, d_res, d_res_converted, cracked_pwd);

    cudaMemcpy(h_cracked_pwd, cracked_pwd, sizeof(unsigned char)*(pwd_len), cudaMemcpyDeviceToHost);

    cudaFree(d_str);
    cudaFree(d_res);

    return h_cracked_pwd;
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

char *target = "de8be12caf23444b451ea27be98dc8a9";


int main() {

	// Set device id to 0 (Use fastest device)
	device.id = 0;
	cudaGetDeviceProperties(&device.prop, device.id);
    getOptimalThreads(&device);

    int passed = 0, failed = 0;

    // run_test("md5(\"\")", hash(""), "d41d8cd98f00b204e9800998ecf8427e") ? passed++ : failed++;
    // run_test("md5(\"a\")", hash("a"), "0cc175b9c0f1b6a831c399e269772661") ? passed++ : failed++;
    // run_test("md5(\"abc\")", md5_hash("abc"), "900150983cd24fb0d6963f7d28e17f72") ? passed++ : failed++;

    // run_test("md5(\"ba\")", md5_hash("1998012"), "de8be12caf23444b451ea27be98dc8a9") ? passed++ : failed++;

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

    char * target_pwd = "a";
    char * target_hash = "0cc175b9c0f1b6a831c399e269772661";
    printf("Cracked pwd = %s\n", md5_hash(target_hash, strlen(target_pwd)));


    return failed;
}
