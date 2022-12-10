#ifndef ___utility___
#define ___utility___

#include <cstdlib>

////////////////////////////////////////////////
//Datatypes                                   //
////////////////////////////////////////////////
#define  uint8_t unsigned char
#define  int32_t int
#define uint32_t unsigned
#define  int64_t long long
#define uint64_t unsigned long long

class uint128_t;
class MD5;
class CSHA1;

////////////////////////////////////////////////
//Helpers                                     //
////////////////////////////////////////////////
//These numbers seem most efficient
//Alternative: 128 Threads, 32 Blocks
#define THREADNUM 64
#define BLOCKNUM  64
#define FIXED      2  //log64(THREADNUM*BLOCKNUM)

////////////////////////////////////////////////
//Unions                                      //
////////////////////////////////////////////////

typedef union{
	uint8_t  val8[8];
	uint32_t val32[2];
	uint64_t val;
} uint64_u;
typedef union{
	uint64_t val64[2];
	uint32_t val32[4];
	uint8_t  val8[16];
} hash_t;


////////////////////////////////////////////////
//Functions                                   //
////////////////////////////////////////////////

void checkCUDAError(const char *msg);

void printHash(const hash_t& h);
__device__ void printHashGPU(hash_t h);
void printSha(const uint8_t* c);
__device__ void printShaGPU(const uint8_t* c);

hash_t parseHash(const char* t);
void parseSha(const char* in, uint8_t* out);

////////////////////////////////////////////////
//Hashing Functions                           //
////////////////////////////////////////////////

__global__ void md5_single(uint8_t* out, const uint8_t* in, int length, int num);
__global__ void sha1_single(uint8_t* out, const uint8_t* in, uint32_t len);

__global__ void testMD5(const char* s, int len);

__device__ void sha1(CSHA1*, uint8_t*, const uint8_t*, uint32_t);
__device__ void sha1(uint8_t*, const uint8_t*, uint32_t);


////////////////////////////////////////////////
//Hashing  Helper Functions                   //
////////////////////////////////////////////////

__device__ void genString (char*, uint128_t, const int, const char*);
__device__ void genPostfix(char*, int, const int, const int, const char*);

#endif
