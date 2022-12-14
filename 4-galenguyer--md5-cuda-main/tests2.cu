#include <cuda.h>
#include <stdint.h>
#include <string.h>
#include "md5.cu"
#include <iostream>
#include <string>

#include <bits/stdc++.h>

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

char *target = "c4ca4238a0b923820dcc509a6f75849b";

void print_str(const char str[],std::string prefix,const int n, const int lenght) {
    if (lenght == 1) {
            for (int j = 0; j < n; j++) {
                if (strcmp(target, md5_hash((prefix + str[j]).c_str())) == 0) {
                    std::cout << "Cracked = " << prefix + str[j] << std::endl;
                    break;
                }
            }
        }
    else {
            for (int i = 0; i < n; i++)
                print_str(str, prefix + str[i], n, lenght - 1);
        }
}


// int main() {
// 	// Set device id to 0 (Use fastest device)
// 	device.id = 0;
// 	cudaGetDeviceProperties(&device.prop, device.id);
//     getOptimalThreads(&device);
//     int len = 2;
//     char str[] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
//     int n = sizeof(str);
//     print_str(str, "", n, len);
//     return failed;
// }
/////////////////
 

std::string getCurrentPwd(int i, char arr[], int len, int n)
{
    std::string result = "";
    for (int j = 0; j < n; j++) {
        result += arr[i % len];
        i /= len;
    }
    return result;
}

int main(int argc, char *argv[]) {

    device.id = 0;
	cudaGetDeviceProperties(&device.prop, device.id);
    getOptimalThreads(&device);

    int n = stoi(argv[1]); // password length
    std::string target = argv[2]; // target hash
    // std::string potential_elements = "0123456789";
    std::string result = "";

    char arr[] = { '0', '1', '2', '3' };
    int len = sizeof(arr) / sizeof(arr[0]);
    for (int i = 0; i < (int)pow(len, n); i++) {
        current_pwd = getCurrentPwd(i, arr, len, n);
        if (strcmp(target, md5_hash(current_pwd).c_str()) == 0) {
            result = current_pwd
            break;
        }
    }
    std::cout << result << std::endl;
}