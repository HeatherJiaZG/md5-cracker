#ifndef ___md5___
#define ___md5___
#include "utility.h"

class MD5{
public:
	friend __device__ void initMd5(MD5*, const uint8_t*, int);
	__device__ MD5(const uint8_t*, int);
	__device__ void get(uint8_t*);
	__device__ const hash_t& getHash();

private:
	__device__ void update(const uint8_t*, int);
	__device__ void transform(const uint8_t*);

	const static uint8_t blocksize = 64;
	uint8_t buffer[blocksize];
	uint64_u count;
	hash_t state;

	__device__ static inline uint32_t F(uint32_t x, uint32_t y, uint32_t z);
	__device__ static inline uint32_t G(uint32_t x, uint32_t y, uint32_t z);
	__device__ static inline uint32_t H(uint32_t x, uint32_t y, uint32_t z);
	__device__ static inline uint32_t I(uint32_t x, uint32_t y, uint32_t z);

	__device__ static inline uint32_t rotate_left(uint32_t, int);

	__device__ static inline void FF(uint32_t &a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t s, uint32_t ac);
	__device__ static inline void GG(uint32_t &a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t s, uint32_t ac);
	__device__ static inline void HH(uint32_t &a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t s, uint32_t ac);
	__device__ static inline void II(uint32_t &a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t s, uint32_t ac);
};
__device__ void initMd5(MD5*, const uint8_t*, int);

void md5_auto(hash_t h, int len);
#endif