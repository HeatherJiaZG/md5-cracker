#ifndef ___bigInt___
#define ___bigInt___
#include "utility.h"

class uint128_t{
public:
	__device__ uint128_t operator+(uint128_t const& rhs);
	__device__ uint128_t operator+(uint64_t  const  rhs);
	__device__ uint128_t operator-(uint128_t const& rhs);
	__device__ uint128_t operator*(const uint32_t rhs);
	__device__ uint128_t operator<< (const uint32_t rhs);
	__device__ uint128_t operator>> (const uint32_t rhs);
	__device__ uint128_t& operator<<=(const uint32_t rhs);
	__device__ uint128_t& operator>>=(const uint32_t rhs);

	__device__ bool      operator<  (const uint128_t& rhs);
	__device__ bool      operator>  (const uint128_t& rhs);

	__device__ uint128_t& operator++ ();

	__device__ void set (const uint64_t lower, const uint64_t upper)
		{val[0]=lower; val[1]=upper;};
	__device__ uint128_t(const uint64_t lower, const uint64_t upper)
		{val[0]=lower; val[1]=upper;};
	__device__ uint128_t(const uint128_t& src)
		{val[0]=src.val[0];val[1]=src.val[1];};
	__device__ uint128_t(const uint64_t v)
		{val[0]=v;         val[1]=0;};

	__device__ uint128_t()
		{val[0]=0;     val[1]=0;};
	uint64_t val[2];
};
__device__ uint128_t bigMul(const uint64_t a, const uint64_t b);

#endif