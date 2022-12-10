#ifndef ___SHA1___
#define ___SHA1___
#include "utility.h"


///////////////////////////////////////////////////////////////////////////
// Declare SHA-1 workspace

typedef union{
	uint8_t c[64];
	uint32_t l[16];
} SHA1_WORKSPACE_BLOCK;

class CSHA1{
public:
	// Constructor and destructor
	__device__ CSHA1();
	friend __device__ void initSHA1(CSHA1*);
	__device__ void Reset();

	// Hash in binary data and strings
	__device__ void Update(const uint8_t* pbData, uint32_t uLen);

	// Finalize hash; call it before using ReportHash(Stl)
	__device__ void Final();

	// Get the raw message digest (20 bytes)
	__device__ void GetHash(uint8_t*) const;
	__device__ const uint8_t* GetPtr() const;

private:
	// Private SHA-1 transformation
	__device__ void Transform(uint32_t* pState, const uint8_t* pBuffer);

	// Member variables
	uint32_t m_state[5];
	uint32_t m_count[2];
	uint8_t m_buffer[64];
	uint8_t m_digest[20];

	uint8_t m_workspace[64];
	SHA1_WORKSPACE_BLOCK* m_block; // SHA1 pointer to the byte array above
};
__device__ void initSHA1(CSHA1*);

void sha1_auto(const uint8_t*, int);
#endif