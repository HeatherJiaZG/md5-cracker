#include "md5.h"
#include "bigInt.h"
#include <stdio.h>
#include <iostream>

#define S11 7
#define S12 12
#define S13 17
#define S14 22
#define S21 5
#define S22 9
#define S23 14
#define S24 20
#define S31 4
#define S32 11
#define S33 16
#define S34 23
#define S41 6
#define S42 10
#define S43 15
#define S44 21


//Helper functions for manipulating states
__device__ inline uint32_t MD5::F(uint32_t x, uint32_t y, uint32_t z) {
	return (x&y) | (~x&z);
}
__device__ inline uint32_t MD5::G(uint32_t x, uint32_t y, uint32_t z) {
	return (x&z) | (y&~z);
}
__device__ inline uint32_t MD5::H(uint32_t x, uint32_t y, uint32_t z) {
	return x^y^z;
}
__device__ inline uint32_t MD5::I(uint32_t x, uint32_t y, uint32_t z) {
	return y ^ (x | ~z);
}
__device__ inline uint32_t MD5::rotate_left(uint32_t x, int n) {
	return (x << n) | (x >> (32-n));
}
__device__ inline void MD5::FF(uint32_t &a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t s, uint32_t ac) {
	a = rotate_left(a+ F(b,c,d) + x + ac, s) + b;
}
__device__ inline void MD5::GG(uint32_t &a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t s, uint32_t ac) {
	a = rotate_left(a + G(b,c,d) + x + ac, s) + b;
}
__device__ inline void MD5::HH(uint32_t &a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t s, uint32_t ac) {
	a = rotate_left(a + H(b,c,d) + x + ac, s) + b;
}
__device__ inline void MD5::II(uint32_t &a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t s, uint32_t ac) {
	a = rotate_left(a + I(b,c,d) + x + ac, s) + b;
}

__constant__ uint8_t padding[64]={
	0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
__device__ void initMd5(MD5* m, const uint8_t* text, int len){
	m->count.val = 0;
	m->state.val64[0] = 0xefcdab8967452301ull;
	m->state.val64[1] = 0x1032547698badcfeull;

	m->update(text, len);

	uint64_u bits = m->count;
	uint32_t index = (m->count.val32[0] >> 3) & 0x3f;
	uint32_t padLen = index < 56 ?
	         56-index : 120 - index;

	m->update(padding, padLen);
	m->update(bits.val8, 8);
}

__device__ MD5::MD5(const uint8_t* text, int len){
	initMd5(this, text, len);
}

__device__ void MD5::transform(const uint8_t *block){
	uint32_t a=state.val32[0], b=state.val32[1],
	         c=state.val32[2], d=state.val32[3];//x[16]

	#define xi(i) x##i = ((uint32_t*)block)[i]
	uint32_t xi( 0), xi( 1), xi( 2), xi( 3), xi( 4),
	         xi( 5), xi( 6), xi( 7), xi( 8), xi( 9),
	         xi(10), xi(11), xi(12), xi(13), xi(14),
	         xi(15);

	//for(int i = 0; i < (blocksize>>2); i++)
	//	x[i] = ((uint32_t*)block)[i];

	/* Round 1 */
	FF (a, b, c, d, x0 , S11, 0xd76aa478); /* 1 */
	FF (d, a, b, c, x1 , S12, 0xe8c7b756); /* 2 */
	FF (c, d, a, b, x2 , S13, 0x242070db); /* 3 */
	FF (b, c, d, a, x3 , S14, 0xc1bdceee); /* 4 */
	FF (a, b, c, d, x4 , S11, 0xf57c0faf); /* 5 */
	FF (d, a, b, c, x5 , S12, 0x4787c62a); /* 6 */
	FF (c, d, a, b, x6 , S13, 0xa8304613); /* 7 */
	FF (b, c, d, a, x7 , S14, 0xfd469501); /* 8 */
	FF (a, b, c, d, x8 , S11, 0x698098d8); /* 9 */
	FF (d, a, b, c, x9 , S12, 0x8b44f7af); /* 10 */
	FF (c, d, a, b, x10, S13, 0xffff5bb1); /* 11 */
	FF (b, c, d, a, x11, S14, 0x895cd7be); /* 12 */
	FF (a, b, c, d, x12, S11, 0x6b901122); /* 13 */
	FF (d, a, b, c, x13, S12, 0xfd987193); /* 14 */
	FF (c, d, a, b, x14, S13, 0xa679438e); /* 15 */
	FF (b, c, d, a, x15, S14, 0x49b40821); /* 16 */

	/* Round 2 */
	GG (a, b, c, d, x1 , S21, 0xf61e2562); /* 17 */
	GG (d, a, b, c, x6 , S22, 0xc040b340); /* 18 */
	GG (c, d, a, b, x11, S23, 0x265e5a51); /* 19 */
	GG (b, c, d, a, x0 , S24, 0xe9b6c7aa); /* 20 */
	GG (a, b, c, d, x5 , S21, 0xd62f105d); /* 21 */
	GG (d, a, b, c, x10, S22,	0x2441453); /* 22 */
	GG (c, d, a, b, x15, S23, 0xd8a1e681); /* 23 */
	GG (b, c, d, a, x4 , S24, 0xe7d3fbc8); /* 24 */
	GG (a, b, c, d, x9 , S21, 0x21e1cde6); /* 25 */
	GG (d, a, b, c, x14, S22, 0xc33707d6); /* 26 */
	GG (c, d, a, b, x3 , S23, 0xf4d50d87); /* 27 */
	GG (b, c, d, a, x8 , S24, 0x455a14ed); /* 28 */
	GG (a, b, c, d, x13, S21, 0xa9e3e905); /* 29 */
	GG (d, a, b, c, x2 , S22, 0xfcefa3f8); /* 30 */
	GG (c, d, a, b, x7 , S23, 0x676f02d9); /* 31 */
	GG (b, c, d, a, x12, S24, 0x8d2a4c8a); /* 32 */

	/* Round 3 */
	HH (a, b, c, d, x5, S31, 0xfffa3942); /* 33 */
	HH (d, a, b, c, x8, S32, 0x8771f681); /* 34 */
	HH (c, d, a, b, x11, S33, 0x6d9d6122); /* 35 */
	HH (b, c, d, a, x14, S34, 0xfde5380c); /* 36 */
	HH (a, b, c, d, x1, S31, 0xa4beea44); /* 37 */
	HH (d, a, b, c, x4, S32, 0x4bdecfa9); /* 38 */
	HH (c, d, a, b, x7, S33, 0xf6bb4b60); /* 39 */
	HH (b, c, d, a, x10, S34, 0xbebfbc70); /* 40 */
	HH (a, b, c, d, x13, S31, 0x289b7ec6); /* 41 */
	HH (d, a, b, c, x0, S32, 0xeaa127fa); /* 42 */
	HH (c, d, a, b, x3, S33, 0xd4ef3085); /* 43 */
	HH (b, c, d, a, x6, S34,	0x4881d05); /* 44 */
	HH (a, b, c, d, x9, S31, 0xd9d4d039); /* 45 */
	HH (d, a, b, c, x12, S32, 0xe6db99e5); /* 46 */
	HH (c, d, a, b, x15, S33, 0x1fa27cf8); /* 47 */
	HH (b, c, d, a, x2, S34, 0xc4ac5665); /* 48 */

	/* Round 4 */
	II (a, b, c, d, x0, S41, 0xf4292244); /* 49 */
	II (d, a, b, c, x7, S42, 0x432aff97); /* 50 */
	II (c, d, a, b, x14, S43, 0xab9423a7); /* 51 */
	II (b, c, d, a, x5, S44, 0xfc93a039); /* 52 */
	II (a, b, c, d, x12, S41, 0x655b59c3); /* 53 */
	II (d, a, b, c, x3, S42, 0x8f0ccc92); /* 54 */
	II (c, d, a, b, x10, S43, 0xffeff47d); /* 55 */
	II (b, c, d, a, x1, S44, 0x85845dd1); /* 56 */
	II (a, b, c, d, x8, S41, 0x6fa87e4f); /* 57 */
	II (d, a, b, c, x15, S42, 0xfe2ce6e0); /* 58 */
	II (c, d, a, b, x6, S43, 0xa3014314); /* 59 */
	II (b, c, d, a, x13, S44, 0x4e0811a1); /* 60 */
	II (a, b, c, d, x4, S41, 0xf7537e82); /* 61 */
	II (d, a, b, c, x11, S42, 0xbd3af235); /* 62 */
	II (c, d, a, b, x2, S43, 0x2ad7d2bb); /* 63 */
	II (b, c, d, a, x9, S44, 0xeb86d391); /* 64 */

	state.val32[0] += a;
	state.val32[1] += b;
	state.val32[2] += c;
	state.val32[3] += d;
}

__device__ void MD5::update(const uint8_t input[], int length){
	int index = (count.val32[0] >> 3) & 0x3f;	// /8 mod 64
	count.val += length << 3;

	int i, firstpart = 64-index;
	if(length >= firstpart){
		memcpy(buffer+index, input, firstpart);
		transform(buffer);
		for(i = firstpart; i+blocksize < length; i+=blocksize)
			transform(input + i);
		index = 0;
	}else
		i=0;
	memcpy(buffer+index, input+i, length-i);
}

__device__ void MD5::get(uint8_t* buf) {
	for(int i = 0; i<16; i++)
		buf[i] = state.val8[i];
	return;
}
__device__ const hash_t& MD5::getHash(){
	return state;
}








static __constant__ hash_t d_h;	//Hash to check, passed by host
static __device__ inline bool checkHash(const hash_t& h){
	return (h.val64[0] == d_h.val64[0])
	     && (h.val64[1] == d_h.val64[1]);
}

//Chars to search
static __shared__ char symbols[64];
static __device__ bool finished = 0;
static __global__ void md5_autoGpu(int len, int devIndex=0, int devNum=1){
	extern __shared__ char share[];

	const int idx = threadIdx.x;
	const int id = idx + blockIdx.x*blockDim.x;
	//myString and myMD5 are in shared memory for performance
	char* myString =    share + idx * len * sizeof(char);
	MD5* myMD5 = (MD5*)(share + len * blockDim.x * sizeof(char)
	                    +sizeof(MD5) * idx);
	if(idx < 64){
		const char sym[] = {
			'a','b','c','d','e','f','g','h','i','j','k','l','m',
			'n','o','p','q','r','s','t','u','v','w','x','y','z',
			'A','B','C','D','E','F','G','H','I','J','K','L','M',
			'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
			'0','1','2','3','4','5','6','7','8','9',' ','-'};
		 symbols[idx] = sym[idx];
	}
	__syncthreads();

	//Will hold the maximum number of iterations needed
	uint128_t max(1);
	//returns number of fixed symbols, currently 3
	genPostfix(myString, id, len, devIndex, symbols);

	//number of iterations 64^(len-fix)
	max <<= 6*(len-FIXED);
	for(uint128_t it; it < max; ++it){
		//generate the prefix for the current iteration
		genString(myString, it, len, symbols);
		//calc MD5
		initMd5(myMD5,(uint8_t*)myString, len);

		//Get the Hash
		bool equal = checkHash(myMD5->getHash());
		if(equal){
			finished=true;
			//Print some information
			printf("ID: %d\n", id);
			printf("Iteration: %016llx%016llx\n", it.val[1],it.val[0]);
			printHashGPU(myMD5->getHash());
			printf("String: ");
			for(int i = 0; i < len; i++)
				printf("%c", myString[i]);
			printf("\n");
		}
		//If anyone found the has found the hash return
		if(finished){
			return;
		}
	}
}


void md5_auto(hash_t h, int len){
	if(len < 2){
		printf("Less than 2 chars not yet supported!");
		exit(-1);
	}

	int devCount;
	cudaGetDeviceCount(&devCount);
	cudaDeviceProp props;

	//We look which device has the most global memory
	//as a good indicator which should be fastest
	size_t maxMem=0, maxId=0;
	for(int i = 0; i < devCount; i++){
		cudaGetDeviceProperties(&props, i);
		if(props.totalGlobalMem > maxMem){
			maxMem = props.totalGlobalMem;
			maxId = i;
		}
	}

	//Set the device and copy the hash
	cudaSetDevice(maxId);
	cudaMemcpyToSymbol(d_h, &h, sizeof(hash_t));
	checkCUDAError("cudaMemcpyToSymbol");

	//Too lazy to add another interface, we just say we had 1 device
	md5_autoGpu<<<BLOCKNUM, THREADNUM,
	              (len+sizeof(MD5))*THREADNUM>>>
	              (len,0,1);
	cudaDeviceSynchronize();
	checkCUDAError("md5_autoGpu");

	/*
	int devNum;
	cudaGetDeviceCount(&devNum);
	cudaStream_t streams[devNum];
	for(int i = 0; i < devNum; i++){
		cudaSetDevice(i);
		cudaStreamCreate(&streams[i]);

		//Copy hash into constant memory
		cudaMemcpyToSymbol(d_h, &h, sizeof(hash_t));
		checkCUDAError("cudaMemcpyToSymbol");

		//We allocate shared memory for each threads string and MD5 object
		md5_autoGpu<<<BLOCKNUM/devNum, THREADNUM,
		              (len+sizeof(MD5))*THREADNUM,streams[i]>>>
		              (len, i, devNum);
	}
	for(int i = 0; i < devNum; i++){
		cudaSetDevice(i);
		cudaDeviceSynchronize();
		checkCUDAError("md5_autoGpu");
	}
	*/
}