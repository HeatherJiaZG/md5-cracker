

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdint.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <device_functions.h>

#include "lib_md5.h"

__device__ void md5_init(struct md5_states* ctx, unsigned char* data, UINT32 len) {
    ctx->a = 0x67452301;
    ctx->b = 0xefcdab89;
    ctx->c = 0x98badcfe;
    ctx->d = 0x10325476;

    for(int i=0; i<14; i++) {
      ctx->in_arr[i] = 0;
    }

    int j = 0;
    for(j=0; j < len; j++){
      int s = (j % 4) * 8;
      UINT32 d = data[j] << s;
      ctx->in_arr[j / 4] |= d;
    }
    int s = (j % 4) * 8;
    UINT32 d = 0x80 << s;
    ctx->in_arr[j / 4] |= d;
    ctx->in_arr[14] = 0;
    ctx->in_arr[15] = len * 8;

    // ctx->count[0] = 0;
    // ctx->count[1] = 0;
    // ctx->count[2] = 0;
    // ctx->count[3] = 0;
}

// __device__ void md5_update(struct md5_states* ctx, UINT32 x, UINT32 y, UINT32 k, UINT32 j) {

//     ctx->k[0] = x;
//     ctx->j[1] = j;
//     ctx->n[2] = k;
//     ctx->m[3] = j;
// }

__device__ inline void md5_run(struct md5_states* ctx){

  UINT32 in[16];
  for(int i=0; i<16; i++) {
    in[i] = ctx->in_arr[i];
  }

  UINT32 a = 0;
  UINT32 b = 0;
  UINT32 c = 0;
  UINT32 d = 0;

  //Initialize hash value for this chunk:
  a = ctx->a;
  b = ctx->b;
  c = ctx->c;
  d = ctx->d;

  // go through 4 rounds
  FF ( a, b, c, d, in[0],  7, 3614090360); 
  FF ( d, a, b, c, in[1],  12, 3905402710); 
  FF ( c, d, a, b, in[2],  17,  606105819); 
  FF ( b, c, d, a, in[3],  22, 3250441966); 
  FF ( a, b, c, d, in[4],  7, 4118548399); 
  FF ( d, a, b, c, in[5],  12, 1200080426);
  FF ( c, d, a, b, in[6],  17, 2821735955); 
  FF ( b, c, d, a, in[7],  22, 4249261313);
  FF ( a, b, c, d, in[8],  7, 1770035416); 
  FF ( d, a, b, c, in[9],  12, 2336552879);
  FF ( c, d, a, b, in[10], 17, 4294925233); 
  FF ( b, c, d, a, in[11], 22, 2304563134);
  FF ( a, b, c, d, in[12], 7, 1804603682); 
  FF ( d, a, b, c, in[13], 12, 4254626195); 
  FF ( c, d, a, b, in[15], 17, 2792965006); 
  FF ( b, c, d, a, in[14], 22, 1236535329); 

  GG ( a, b, c, d, in[1], 5, 4129170786);
  GG ( d, a, b, c, in[6], 9, 3225465664); 
  GG ( c, d, a, b, in[11], 14,  643717713); 
  GG ( b, c, d, a, in[0], 20, 3921069994); 
  GG ( a, b, c, d, in[5], 5, 3593408605); 
  GG ( d, a, b, c, in[10], 9,   38016083); 
  GG ( c, d, a, b, in[14], 14, 3634488961);
  GG ( b, c, d, a, in[4], 20, 3889429448); 
  GG ( a, b, c, d, in[9], 5,  568446438); 
  GG ( d, a, b, c, in[15], 9, 3275163606); 
  GG ( c, d, a, b, in[3], 14, 4107603335); 
  GG ( b, c, d, a, in[8], 20, 1163531501); 
  GG ( a, b, c, d, in[13], 5, 2850285829);
  GG ( d, a, b, c, in[2], 9, 4243563512); 
  GG ( c, d, a, b, in[7], 14, 1735328473);
  GG ( b, c, d, a, in[12], 20, 2368359562); 

  HH ( a, b, c, d, in[5], 4, 4294588738); 
  HH ( d, a, b, c, in[8], 11, 2272392833);
  HH ( c, d, a, b, in[11], 16, 1839030562); 
  HH ( b, c, d, a, in[15], 23, 4259657740);
  HH ( a, b, c, d, in[1], 4, 2763975236); 
  HH ( d, a, b, c, in[4], 11, 1272893353); 
  HH ( c, d, a, b, in[7], 16, 4139469664); 
  HH ( b, c, d, a, in[10], 23, 3200236656); 
  HH ( a, b, c, d, in[13], 4,  681279174);
  HH ( d, a, b, c, in[0], 11, 3936430074); 
  HH ( c, d, a, b, in[3], 16, 3572445317); 
  HH ( b, c, d, a, in[6], 23,   76029189); 
  HH ( a, b, c, d, in[9], 4, 3654602809); 
  HH ( d, a, b, c, in[12], 11, 3873151461); 
  HH ( c, d, a, b, in[14], 16,  530742520); 
  HH ( b, c, d, a, in[2], 23, 3299628645); 

  II ( a, b, c, d, in[0], 6, 4096336452); 
  II ( d, a, b, c, in[7], 10, 1126891415); 
  II ( c, d, a, b, in[15], 15, 2878612391); 
  II ( b, c, d, a, in[5], 21, 4237533241); 
  II ( a, b, c, d, in[12], 6, 1700485571); 
  II ( d, a, b, c, in[3], 10, 2399980690); 
  II ( c, d, a, b, in[10], 15, 4293915773); 
  II ( b, c, d, a, in[1], 21, 2240044497); 
  II ( a, b, c, d, in[8], 6, 1873313359); 
  II ( d, a, b, c, in[14], 10, 4264355552); 
  II ( c, d, a, b, in[6], 15, 2734768916); 
  II ( b, c, d, a, in[13], 21, 1309151649); 
  II ( a, b, c, d, in[4], 6, 4149444226); 
  II ( d, a, b, c, in[11], 10, 3174756917); 
  II ( c, d, a, b, in[2], 15,  718787259);
  II ( b, c, d, a, in[9], 21, 3951481745); 

  a += ctx->a;
  b += ctx->b;
  c += ctx->c;
  d += ctx->d;

  ctx->hashes[0] = a;
  ctx->hashes[1] = b;
  ctx->hashes[2] = c;
  ctx->hashes[3] = d;
}