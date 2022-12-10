#include <iostream>
#include "md5.h"
#include "SHA1.h"
#include <cstring>
#include <cstdio>

static void cudaTest(const char* s, int l, const char* h){
	char* d_s;
	cudaMalloc(&d_s,   (l+1)*sizeof(char));
	cudaMemcpy(d_s, s, (l+1)*sizeof(char), cudaMemcpyHostToDevice);

	testMD5<<<1,1>>>(d_s, l);
	cudaDeviceSynchronize();
	checkCUDAError("Kernel launch");

	cudaFree(d_s);
	std::cout << h <<"\n"<< std::endl;
}


int doMd5(int argc, char** argv){
	//Hash single string
	int len = strlen(argv[1]);

	uint8_t *d_s, *d_out, h_out[33];
	cudaMalloc(&d_s, (len+1)*sizeof(char));
	cudaMalloc(&d_out, 16*sizeof(char));

	cudaMemcpy(d_s, argv[1], (len+1)*sizeof(char), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpyHostToDevice");

	md5_single<<<1,1>>>(d_out, d_s, len,  1);

	cudaMemcpy(h_out, d_out, 16*sizeof(char), cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpyDeviceToHost");

	hash_t h;
	for(int i = 0; i < 16; i++)
		h.val8[i] = h_out[i];
	printHash(h);
	return 0;
}

int main(int argc, char** argv){
	if(argc < 2)
		return -1;
	if(!strcmp(argv[1],"test")){
		cudaTest("",
		         0,"d41d8cd98f00b204e9800998ecf8427e");
		cudaTest("a",
		         1,"0cc175b9c0f1b6a831c399e269772661");
		cudaTest("abc",
		         3,"900150983cd24fb0d6963f7d28e17f72");
		cudaTest("message digest",
		         14,"f96b697d7cb7938d525a2f31aaf161d0");
		cudaTest("abcdefghijklmnopqrstuvwxyz",
		         26,"c3fcd3d76192e4007dfb496cca67e13b");
		cudaTest("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
		         62,"d174ab98d277d9f5a5611c2c9f419d9f");
		cudaTest("12345678901234567890123456789012345678901234567890123456789012345678901234567890",
		         80,"57edf4a22be3c955ac49da2e2107b67a");
		return 0;
	}
	if(argc==4){
		int len = atoi(argv[2]);

		if(!strcmp(argv[1], "md5")){
			//Find the hash
			hash_t h = parseHash(argv[3]);
			printHash(h);
			md5_auto(h, len);
			return 0;
		} else if(!strcmp(argv[1],"sha1")){
			uint8_t hash[20];
			parseSha(argv[3], hash);

			printSha(hash);
			sha1_auto(hash, len);	//TODO implement
			return 0;
		}else{
			printf("First argument must be md5 or sha1");
			return -1;
		}
	}
	if(argc == 3){
		if(!strcmp(argv[1], "md5")){
			return doMd5(argc-1, argv+1);
		} else if(!strcmp(argv[1],"sha1")){
			int len = strlen(argv[2]);
			uint8_t out[20];
			uint8_t *d_in, *d_out;

			cudaMalloc(&d_in, len+1);
			cudaMalloc(&d_out, 20);

			cudaMemcpy(d_in, argv[2], len+1, cudaMemcpyHostToDevice);
			sha1_single<<<1,1>>>(d_out, d_in, len);
			cudaDeviceSynchronize();
			checkCUDAError("Kernel launch");

			cudaMemcpy(out, d_out, 20, cudaMemcpyDeviceToHost);
			printSha(out);
			return 0;
		}else{
			printf("First argument must be md5 or sha1");
		}
	}
	if(argc==2){
		return doMd5(argc, argv);
	}
	printf("Usage:\n");
	printf("./main {md5, sha1} {String}\n");
	printf("./main {md5, sha1} {Number characters} {Hash}\n");
}
