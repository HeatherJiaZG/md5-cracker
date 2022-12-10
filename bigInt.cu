#include "bigInt.h"

__device__ uint128_t uint128_t::operator+(uint128_t const& rhs){
	uint128_t x;
	//We use asm here to utilize the carry flag
	asm("add.cc.u64 %0, %2, %4;"        //Add lower with carry-out
		"addc.u64   %1, %3, %5;"        //Add upper with carry-in
		:"=l"(x.val[0])   , "=l"(x.val[1])   //0,1=result
		: "l"(val[0])     ,  "l"(val[1])      //2,3=lhs
		, "l"(rhs.val[0]) ,  "l"(rhs.val[1]));//4,5=rhs
	return x;
}
__device__ uint128_t uint128_t::operator+(uint64_t  const  rhs){
	uint128_t x;
	//We use asm here to utilize the carry flag
	asm("add.cc.u64 %0, %2, %4;"        //Add lower with carry-out
		"addc.u64   %1, %3, 0;"        //Add upper with carry-in
		:"=l"(x.val[0])   , "=l"(x.val[1])   //0,1=result
		: "l"(val[0])     ,  "l"(val[1])      //2,3=lhs
		, "l"(rhs));//4=rhs
	return x;
}

__device__ uint128_t uint128_t::operator-(uint128_t const& rhs){
	uint128_t x;
	//same as above
	asm("sub.cc.u64 %0, %2, %4;"
		"subc.u64   %1, %3, %5;"
		:"=l"(x.val[0])   , "=l"(x.val[1])   //0,1=result
		: "l"(val[0])     ,  "l"(val[1])      //2,3=lhs
		, "l"(rhs.val[0]) ,  "l"(rhs.val[1]));//4,5=rhs)
	return x;
}


__device__ uint128_t uint128_t::operator<<(const uint32_t rhs){
	uint128_t x(*this);
	return x <<= rhs;
}
__device__ uint128_t uint128_t::operator>>(const uint32_t rhs){
	uint128_t x(*this);
	return x >>= rhs;
}

__device__ uint128_t& uint128_t::operator<<=(const uint32_t rhs){
	val[1] <<= rhs;
	val[1] |=  val[0] >> 64-rhs;
	val[0] <<= rhs;
	return *this;
}
__device__ uint128_t& uint128_t::operator>>=(const uint32_t rhs){
	val[0] >>= rhs;
	val[0] |=  val[1] << 64-rhs;
	val[1] >>= rhs;
	return *this;
}

__device__ uint128_t uint128_t::operator*(const uint32_t rhs){
	uint128_t x;
	x = bigMul(val[0], rhs);          //calculate lower and carry
	x.val[1] = val[1]*rhs + x.val[1]; //upper+carry
	return x;
};

__device__ bool uint128_t::operator<  (const uint128_t& rhs){
	return rhs.val[1] > this->val[1]
	        || (this->val[1] == rhs.val[1]
	           && this->val[0] < rhs.val[0]);
}
__device__ bool uint128_t::operator>  (const uint128_t& rhs){
	return this->val[1] > rhs.val[1]
	        || (this->val[1] == rhs.val[1]
	           && this->val[0] > rhs.val[0]);
}

__device__ uint128_t& uint128_t::operator++(){
	*this = *this+1;
	return *this;
}

__device__ uint128_t bigMul(const uint64_t a, const uint64_t b){
	uint128_t x;
	asm("mul.lo.u64 %0, %2, %3;"    //multiply lower
		"mul.hi.u64 %1, %2, %3;"    //multiply upper
		:"=l"(x.val[0]) , "=l"(x.val[1])//0,1=result
		:"l" (a) , "l" (b));   //2,3=a,b
	return x;
}
