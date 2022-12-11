#ifndef __MD5LIBGPU_H__
#define __MD5LIBGPU_H__

#include <stdio.h>
#include <sys/types.h>
#include "md5CrackerGPU.h"

int callMD5CUDA(struct deviceInfo *,char *, int *,int *, int,int *);

#endif
