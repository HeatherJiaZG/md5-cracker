

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdint.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <device_functions.h>

#include "lib_md5.h"

__device__ void md5_init(struct md5_context* ctx, unsigned char* data, uint32_t length) {
    ctx->a = 0x67452301;
    ctx->b = 0xefcdab89;
    ctx->c = 0x98badcfe;
    ctx->d = 0x10325476;

    for(int i=0; i<14; i++) {
      ctx->in_arr[i] = 0;
    }

    int j = 0;
    for(j=0; j < length; j++){
      int s = (j % 4) * 8;
      uint32_t d = data[j] << s;
      ctx->in_arr[j / 4] |= d;
    }
    int s = (j % 4) * 8;
    uint32_t d = 0x80 << s;
    ctx->in_arr[j / 4] |= d;
    ctx->in_arr[14] = 0;
    ctx->in_arr[15] = length * 8;

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

__device__ inline void md5Hash(struct md5_context* ctx){

  uint32_t in[16];
  for(int i=0; i<16; i++) {
    in[i] = ctx->in_arr[i];
  }

  uint32_t a = 0;
  uint32_t b = 0;
  uint32_t c = 0;
  uint32_t d = 0;

  //Initialize hash value for this chunk:
  a = ctx->a;
  b = ctx->b;
  c = ctx->c;
  d = ctx->d;

  FF ( a, b, c, d, in[0],  7, 3614090360); /* 1 */
  FF ( d, a, b, c, in[1],  12, 3905402710); /* 2 */
  FF ( c, d, a, b, in[2],  17,  606105819); /* 3 */
  FF ( b, c, d, a, in[3],  22, 3250441966); /* 4 */
  FF ( a, b, c, d, in[4],  7, 4118548399); /* 5 */
  FF ( d, a, b, c, in[5],  12, 1200080426); /* 6 */
  FF ( c, d, a, b, in[6],  17, 2821735955); /* 7 */
  FF ( b, c, d, a, in[7],  22, 4249261313); /* 8 */
  FF ( a, b, c, d, in[8],  7, 1770035416); /* 9 */
  FF ( d, a, b, c, in[9],  12, 2336552879); /* 10 */
  FF ( c, d, a, b, in[10], 17, 4294925233); /* 11 */
  FF ( b, c, d, a, in[11], 22, 2304563134); /* 12 */
  FF ( a, b, c, d, in[12], 7, 1804603682); /* 13 */
  FF ( d, a, b, c, in[13], 12, 4254626195); /* 14 */
  FF ( c, d, a, b, in[15], 17, 2792965006); /* 15 */
  FF ( b, c, d, a, in[14], 22, 1236535329); /* 16 */

  GG ( a, b, c, d, in[1], 5, 4129170786); /* 17 */
  GG ( d, a, b, c, in[6], 9, 3225465664); /* 18 */
  GG ( c, d, a, b, in[11], 14,  643717713); /* 19 */
  GG ( b, c, d, a, in[0], 20, 3921069994); /* 20 */
  GG ( a, b, c, d, in[5], 5, 3593408605); /* 21 */
  GG ( d, a, b, c, in[10], 9,   38016083); /* 22 */
  GG ( c, d, a, b, in[14], 14, 3634488961); /* 23 */
  GG ( b, c, d, a, in[4], 20, 3889429448); /* 24 */
  GG ( a, b, c, d, in[9], 5,  568446438); /* 25 */
  GG ( d, a, b, c, in[15], 9, 3275163606); /* 26 */
  GG ( c, d, a, b, in[3], 14, 4107603335); /* 27 */
  GG ( b, c, d, a, in[8], 20, 1163531501); /* 28 */
  GG ( a, b, c, d, in[13], 5, 2850285829); /* 29 */
  GG ( d, a, b, c, in[2], 9, 4243563512); /* 30 */
  GG ( c, d, a, b, in[7], 14, 1735328473); /* 31 */
  GG ( b, c, d, a, in[12], 20, 2368359562); /* 32 */

  HH ( a, b, c, d, in[5], 4, 4294588738); /* 33 */
  HH ( d, a, b, c, in[8], 11, 2272392833); /* 34 */
  HH ( c, d, a, b, in[11], 16, 1839030562); /* 35 */
  HH ( b, c, d, a, in[15], 23, 4259657740); /* 36 */
  HH ( a, b, c, d, in[1], 4, 2763975236); /* 37 */
  HH ( d, a, b, c, in[4], 11, 1272893353); /* 38 */
  HH ( c, d, a, b, in[7], 16, 4139469664); /* 39 */
  HH ( b, c, d, a, in[10], 23, 3200236656); /* 40 */
  HH ( a, b, c, d, in[13], 4,  681279174); /* 41 */
  HH ( d, a, b, c, in[0], 11, 3936430074); /* 42 */
  HH ( c, d, a, b, in[3], 16, 3572445317); /* 43 */
  HH ( b, c, d, a, in[6], 23,   76029189); /* 44 */
  HH ( a, b, c, d, in[9], 4, 3654602809); /* 45 */
  HH ( d, a, b, c, in[12], 11, 3873151461); /* 46 */
  HH ( c, d, a, b, in[14], 16,  530742520); /* 47 */
  HH ( b, c, d, a, in[2], 23, 3299628645); /* 48 */

  II ( a, b, c, d, in[0], 6, 4096336452); /* 49 */
  II ( d, a, b, c, in[7], 10, 1126891415); /* 50 */
  II ( c, d, a, b, in[15], 15, 2878612391); /* 51 */
  II ( b, c, d, a, in[5], 21, 4237533241); /* 52 */
  II ( a, b, c, d, in[12], 6, 1700485571); /* 53 */
  II ( d, a, b, c, in[3], 10, 2399980690); /* 54 */
  II ( c, d, a, b, in[10], 15, 4293915773); /* 55 */
  II ( b, c, d, a, in[1], 21, 2240044497); /* 56 */
  II ( a, b, c, d, in[8], 6, 1873313359); /* 57 */
  II ( d, a, b, c, in[14], 10, 4264355552); /* 58 */
  II ( c, d, a, b, in[6], 15, 2734768916); /* 59 */
  II ( b, c, d, a, in[13], 21, 1309151649); /* 60 */
  II ( a, b, c, d, in[4], 6, 4149444226); /* 61 */
  II ( d, a, b, c, in[11], 10, 3174756917); /* 62 */
  II ( c, d, a, b, in[2], 15,  718787259); /* 63 */
  II ( b, c, d, a, in[9], 21, 3951481745); /* 64 */

  a += ctx->a;
  b += ctx->b;
  c += ctx->c;
  d += ctx->d;

  ctx->threadHash[0] = a;
  ctx->threadHash[1] = b;
  ctx->threadHash[2] = c;
  ctx->threadHash[3] = d;
}