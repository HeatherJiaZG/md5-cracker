#include <cstring>
#include <cstdio>
#include "SHA1.h"

static void printHash(unsigned char* c){
	printf("Hash: %02x%02x%02x%02x%02x%02x%02x%02x%02x%02x"
		         "%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x\n",
        c[0 ],c[1 ],c[2 ],c[3 ],c[4 ],c[5 ],
        c[6 ],c[7 ],c[8 ],c[9 ],c[10],c[11],
        c[12],c[13],c[14],c[15],c[16],c[17],
        c[18],c[19]);

}

int main(int argc, char const *argv[]){
	if(argc<2)
		return -1;
	CSHA1 sha;
	sha.Update((unsigned char*)(argv[1]), strlen(argv[1]));
	sha.Final();

	unsigned char r[20];
	sha.GetHash(r);
	printHash(r);

	return 0;
}