#ifndef __MD5LIBGPU_H__
#define __MD5LIBGPU_H__

#include <stdio.h>
#include <sys/types.h>

int callMD5CUDA(struct deviceInfo *,char *, int *,int *, int,int *);

#endif
