#include "utility.h"

#include <iostream>
#include <cstdio>
#include "md5.h"
#include "SHA1.h"
#include "bigInt.h"

////////////////////////////////////////////////
//Functions                                   //
////////////////////////////////////////////////

void checkCUDAError(const char *msg){
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err){
		std::cerr << "Cuda error: " << msg << ": "
			<< cudaGetErrorString(err) << ".\n";
		exit(EXIT_FAILURE);
	}
}

void printHash(const hash_t& h){
	printf("Hash: %02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x\n",
	    h.val8[0],h.val8[1],h.val8[2],h.val8[3],h.val8[4],h.val8[5],h.val8[6],h.val8[7]
	   ,h.val8[8],h.val8[9],h.val8[10],h.val8[11],h.val8[12],h.val8[13],h.val8[14],h.val8[15]);
}
__device__ void printHashGPU(hash_t h){
	printf("Hash: %02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x\n",
	  h.val8[0],h.val8[1],h.val8[2],h.val8[3],h.val8[4],h.val8[5],h.val8[6],h.val8[7]
	 ,h.val8[8],h.val8[9],h.val8[10],h.val8[11],h.val8[12],h.val8[13],h.val8[14],h.val8[15]);
}


void printSha(const uint8_t* c){
	printf("Hash: %02x%02x%02x%02x%02x%02x%02x%02x%02x%02x"
		         "%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x\n",
	    c[0 ],c[1 ],c[2 ],c[3 ],c[4 ],c[5 ],
	    c[6 ],c[7 ],c[8 ],c[9 ],c[10],c[11],
	    c[12],c[13],c[14],c[15],c[16],c[17],
	    c[18],c[19]);
}
__device__ void printShaGPU(const uint8_t* c){
	printf("Hash: %02x%02x%02x%02x%02x%02x%02x%02x%02x%02x"
		         "%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x\n",
	    c[0 ],c[1 ],c[2 ],c[3 ],c[4 ],c[5 ],
	    c[6 ],c[7 ],c[8 ],c[9 ],c[10],c[11],
	    c[12],c[13],c[14],c[15],c[16],c[17],
	    c[18],c[19]);
}

hash_t parseHash(const char* t){
	hash_t h;
	char h_string[3];
	h_string[2] = 0;
	//Parse Hash
	for(int j = 0; j < 16; j++){
		for(int i = 0; i < 2; i++)
			h_string[i] = t[i+j*2];
		h.val8[j] = strtol(h_string,NULL, 16);
	}
	return h;
}
void parseSha(const char* in, uint8_t* out){
	char h_string[3];
	h_string[2] = 0;

	for(int j = 0; j < 20; j++){
		for(int i = 0; i < 2; i++)
			h_string[i] = in[i+2*j];
		out[j] = strtol(h_string, NULL, 16);
	}
}


////////////////////////////////////////////////
//Hashing Functions                           //
////////////////////////////////////////////////

//Hash a single string
__global__ void md5_single(uint8_t* output, const uint8_t* input, int length, int number){
	MD5 md(input, length);
	md.get(output);
}

__device__ void sha1(CSHA1* sha,uint8_t* out, const uint8_t* in, uint32_t len){
	initSHA1(sha);
	sha->Update(in, len);
	sha->Final();
	sha->GetHash(out);
}
__device__ void sha1(uint8_t* out, const uint8_t* in, uint32_t len){
	CSHA1 sha;
	sha1(&sha, out, in, len);
}
__global__ void sha1_single(uint8_t* out, const uint8_t* in, uint32_t len){
	sha1(out, in, len);
}


//Hash a single string and print the hash
__global__ void testMD5(const char* s, int len){
	MD5 md((unsigned char*)s, len);
	printHashGPU(md.getHash());
}

////////////////////////////////////////////////
//Hashing  Helper Functions                   //
////////////////////////////////////////////////

//generate the string depending on ID and round
__device__ void genString(char* target, uint128_t it, const int len, const char* symbols){
	#pragma unroll 4
	for(int i = 0; i < len-FIXED; it >>= 6, ++i)
		target[i] = symbols[it.val[0]&63];
}

//Generate the string postfix
//Returns the remaining number of iterations
__device__ void genPostfix(char* target, int id, const int len, const int devIndex, const char* symbols){
	//We have exactly 3 fixed symbols, determined by the id
	id += devIndex*blockDim.x*gridDim.x;
	#pragma unroll 2
	for(int i = len-FIXED; i < len; id>>=6, i++)
		target[i] = symbols[id & 63];
}