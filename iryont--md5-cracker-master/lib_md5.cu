

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdint.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <device_functions.h>

#include "lib_md5.h"

__device__ void md5_init(struct md5_context* ctx) {
    ctx->a = 0x67452301;
    ctx->b = 0xefcdab89;
    ctx->c = 0x98badcfe;
    ctx->d = 0x10325476;

    // ctx->count[0] = 0;
    // ctx->count[1] = 0;
    // ctx->count[2] = 0;
    // ctx->count[3] = 0;
}

// __device__ void md5_update(struct md5_context* ctx, uint32_t a1, uint32_t b1, uint32_t c1, uint32_t d1) {

//     ctx->k[0] = a1;
//     ctx->j[1] = b1;
//     ctx->n[2] = c1;
//     ctx->m[3] = d1;
// }

__device__ inline void md5Hash(struct md5_context* ctx, unsigned char* data, uint32_t length){

  uint32_t a = 0;
  uint32_t b = 0;
  uint32_t c = 0;
  uint32_t d = 0;

  uint32_t in[14] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0};

  int i = 0;
  for(i=0; i < length; i++){
    in[i / 4] |= data[i] << ((i % 4) * 8);
  }
  
  in[i / 4] |= 0x80 << ((i % 4) * 8);

  uint32_t bitlen = length * 8;

  // #define in0  (vals[0])//x
  // #define in1  (vals[1])//y
  // #define in2  (vals[2])//z
  // #define in3  (vals[3])
  // #define in4  (vals[4])
  // #define in5  (vals[5])
  // #define in6  (vals[6])
  // #define in7  (vals[7])
  // #define in8  (vals[8])
  // #define in9  (vals[9])
  // #define in10 (vals[10])
  // #define in11 (vals[11])
  // #define in12 (vals[12])
  // #define in13 (vals[13])
  // #define in14 (bitlen) //w = bit length
  // #define in15 (0)

  //Initialize hash value for this chunk:
  a = ctx->a;
  b = ctx->b;
  c = ctx->c;
  d = ctx->d;

  /* Round 1 */
  #define S11 7
  #define S12 12
  #define S13 17
  #define S14 22
  FF ( a, b, c, d, in[0],  S11, 3614090360); /* 1 */
  FF ( d, a, b, c, in[1],  S12, 3905402710); /* 2 */
  FF ( c, d, a, b, in[2],  S13,  606105819); /* 3 */
  FF ( b, c, d, a, in[3],  S14, 3250441966); /* 4 */
  FF ( a, b, c, d, in[4],  S11, 4118548399); /* 5 */
  FF ( d, a, b, c, in[5],  S12, 1200080426); /* 6 */
  FF ( c, d, a, b, in[6],  S13, 2821735955); /* 7 */
  FF ( b, c, d, a, in[7],  S14, 4249261313); /* 8 */
  FF ( a, b, c, d, in[8],  S11, 1770035416); /* 9 */
  FF ( d, a, b, c, in[9],  S12, 2336552879); /* 10 */
  FF ( c, d, a, b, in[10], S13, 4294925233); /* 11 */
  FF ( b, c, d, a, in[11], S14, 2304563134); /* 12 */
  FF ( a, b, c, d, in[12], S11, 1804603682); /* 13 */
  FF ( d, a, b, c, in[13], S12, 4254626195); /* 14 */
  FF ( c, d, a, b, bitlen, S13, 2792965006); /* 15 */
  FF ( b, c, d, a, 0, S14, 1236535329); /* 16 */

  /* Round 2 */
  #define S21 5
  #define S22 9
  #define S23 14
  #define S24 20
  GG ( a, b, c, d, in[1], S21, 4129170786); /* 17 */
  GG ( d, a, b, c, in[6], S22, 3225465664); /* 18 */
  GG ( c, d, a, b, in[11], S23,  643717713); /* 19 */
  GG ( b, c, d, a, in[0], S24, 3921069994); /* 20 */
  GG ( a, b, c, d, in[5], S21, 3593408605); /* 21 */
  GG ( d, a, b, c, in[10], S22,   38016083); /* 22 */
  GG ( c, d, a, b, 0, S23, 3634488961); /* 23 */
  GG ( b, c, d, a, in[4], S24, 3889429448); /* 24 */
  GG ( a, b, c, d, in[9], S21,  568446438); /* 25 */
  GG ( d, a, b, c, bitlen, S22, 3275163606); /* 26 */
  GG ( c, d, a, b, in[3], S23, 4107603335); /* 27 */
  GG ( b, c, d, a, in[8], S24, 1163531501); /* 28 */
  GG ( a, b, c, d, in[13], S21, 2850285829); /* 29 */
  GG ( d, a, b, c, in[2], S22, 4243563512); /* 30 */
  GG ( c, d, a, b, in[7], S23, 1735328473); /* 31 */
  GG ( b, c, d, a, in[12], S24, 2368359562); /* 32 */

  /* Round 3 */
  #define S31 4
  #define S32 11
  #define S33 16
  #define S34 23
  HH ( a, b, c, d, in[5], S31, 4294588738); /* 33 */
  HH ( d, a, b, c, in[8], S32, 2272392833); /* 34 */
  HH ( c, d, a, b, in[11], S33, 1839030562); /* 35 */
  HH ( b, c, d, a, bitlen, S34, 4259657740); /* 36 */
  HH ( a, b, c, d, in[1], S31, 2763975236); /* 37 */
  HH ( d, a, b, c, in[4], S32, 1272893353); /* 38 */
  HH ( c, d, a, b, in[7], S33, 4139469664); /* 39 */
  HH ( b, c, d, a, in[10], S34, 3200236656); /* 40 */
  HH ( a, b, c, d, in[13], S31,  681279174); /* 41 */
  HH ( d, a, b, c, in[0], S32, 3936430074); /* 42 */
  HH ( c, d, a, b, in[3], S33, 3572445317); /* 43 */
  HH ( b, c, d, a, in[6], S34,   76029189); /* 44 */
  HH ( a, b, c, d, in[9], S31, 3654602809); /* 45 */
  HH ( d, a, b, c, in[12], S32, 3873151461); /* 46 */
  HH ( c, d, a, b, 0, S33,  530742520); /* 47 */
  HH ( b, c, d, a, in[2], S34, 3299628645); /* 48 */

  /* Round 4 */
  #define S41 6
  #define S42 10
  #define S43 15
  #define S44 21
  II ( a, b, c, d, in[0], S41, 4096336452); /* 49 */
  II ( d, a, b, c, in[7], S42, 1126891415); /* 50 */
  II ( c, d, a, b, bitlen, S43, 2878612391); /* 51 */
  II ( b, c, d, a, in[5], S44, 4237533241); /* 52 */
  II ( a, b, c, d, in[12], S41, 1700485571); /* 53 */
  II ( d, a, b, c, in[3], S42, 2399980690); /* 54 */
  II ( c, d, a, b, in[10], S43, 4293915773); /* 55 */
  II ( b, c, d, a, in[1], S44, 2240044497); /* 56 */
  II ( a, b, c, d, in[8], S41, 1873313359); /* 57 */
  II ( d, a, b, c, 0, S42, 4264355552); /* 58 */
  II ( c, d, a, b, in[6], S43, 2734768916); /* 59 */
  II ( b, c, d, a, in[13], S44, 1309151649); /* 60 */
  II ( a, b, c, d, in[4], S41, 4149444226); /* 61 */
  II ( d, a, b, c, in[11], S42, 3174756917); /* 62 */
  II ( c, d, a, b, in[2], S43,  718787259); /* 63 */
  II ( b, c, d, a, in[9], S44, 3951481745); /* 64 */

  a += ctx->a;
  b += ctx->b;
  c += ctx->c;
  d += ctx->d;

  ctx->threadHash[0] = a;
  ctx->threadHash[1] = b;
  ctx->threadHash[2] = c;
  ctx->threadHash[3] = d;
}