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

/* typedef a 32 bit type */
typedef unsigned int UINT4;

/* Data structure for MD5 (Message Digest) computation */
typedef struct {
  UINT4 i[2];                   /* number of _bits_ handled mod 2^64 */
  UINT4 buf[4];                                    /* scratch buffer */
  unsigned char in[64];                              /* input buffer */
  unsigned char digest[16];     /* actual digest after MD5Final call */
} MD5_CTX;

extern float totalTime;

// __device__ void MD5Init (MD5_CTX * mdContext);
// __device__ void MD5Update (MD5_CTX * mdContext, unsigned char * inBuf, unsigned int inLen);
// __device__ void MD5Final (MD5_CTX * mdContext);
extern int callMD5CUDA(struct deviceInfo *,char *, int *,int *, int,int *);

#endif
