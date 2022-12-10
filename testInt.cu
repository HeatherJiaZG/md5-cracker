#include "stdio.h"
#include <iostream>
#include "bigInt.h"

#define OP(x) (++(x))

__device__ void print(uint128_t& u){
		printf("%016llx%016llx\n", u.val[1], u.val[0]);

}
__device__ void print(const unsigned int i, uint128_t u){
		printf("%3u: %016llx%016llx\n",i, u.val[1], u.val[0]);
}

__global__ void testInt(uint128_t* in, uint128_t* out){
	const unsigned int& idx = threadIdx.x;
	if(idx == 0){
		in[0].set(1,0);
		in[1].set(0,1);
		in[2].set(256,0);
		in[3].set(0,256);
		in[4].set(0xFFFFFFFFu, 0);           //max int
		in[5].set(0, 0xFFFFFFFFu);
		in[6].set(0xFFFFFFFFFFFFFFFFull, 0); //max long
		in[7].set(0, 0xFFFFFFFFFFFFFFFFull);
		printf("Initial:\n");
	}
	print(in[idx]);
	print(idx, OP(in[idx]));
	__syncthreads();
}

int main(int argc, char *argv[]){
	uint128_t *d_in, *d_out;
	cudaMalloc(&d_in, 8*sizeof(uint128_t));
	cudaMalloc(&d_out, 8*sizeof(uint128_t));
	testInt<<<1,8>>>(d_in, d_out);
	cudaDeviceSynchronize();

	unsigned __int128 h1, h2;
	h1  = 0xFFFFFFFFFFFFFFFFull;
	h2 = 0xFFFFFFFFFFFFFFFFull;
	h1 <<= 64;
	printf("     %016llx%016llx\n", (uint64_t) (OP(h2) >> 64), (uint64_t) OP(h2));
	printf("     %016llx%016llx\n", (uint64_t) (OP(h1) >> 64), (uint64_t) OP(h1));
	return 0;
}