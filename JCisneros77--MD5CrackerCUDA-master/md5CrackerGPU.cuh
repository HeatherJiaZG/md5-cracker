#ifndef __MD5CRACKERGPU_H__
#define __MD5CRACKERGPU_H__

// Struct to store device information
struct deviceInfo{
	struct cudaDeviceProp prop; // Device Properties
	int id; // Device ID
	int max_threads; // Device max threads per block
	int max_blocks; // Device max blocks
	int global_memory_len;
};

float totalTime;

__device__ void MD5Init (MD5_CTX * mdContext);
__device__ void MD5Update (MD5_CTX * mdContext, unsigned char * inBuf, unsigned int inLen);
__device__ void MD5Final (MD5_CTX * mdContext);
int callMD5CUDA(struct deviceInfo *,char *, int *,int *, int,int *);

#endif