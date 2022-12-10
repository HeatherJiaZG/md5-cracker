#include <iostream>
#include "md5.h"
#include <string.h>

using std::cout; using std::endl;

typedef union{
	unsigned char      val8[16];
	unsigned short     val16[8];
	unsigned int       val32[4];
	unsigned long long val64[2];
} hash_t;


hash_t parseHash(const char* t){
	hash_t h;
	char h_string[3];
	h_string[2] = 0;
	//Parse Hash
	for(int j = 0; j < 16; j++){
		for(int i = 0; i < 2; i++)
			h_string[i] = t[i+j*2];
		h.val8[j] = strtol(h_string,NULL, 16);
	}
	return h;
}

const char symbols[] = {
	'a','b','c','d','e','f','g','h','i','j','k','l','m',
	'n','o','p','q','r','s','t','u','v','w','x','y','z',
	'A','B','C','D','E','F','G','H','I','J','K','L','M',
	'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
	'0','1','2','3','4','5','6','7','8','9',' ','-'};

static inline void genString(char* target, unsigned long long it, const int len){
	for(int i = 0; i < len; it >>= 6, ++i)
		target[i] = symbols[it&63];
}

int main(int argc, char *argv[])
{
	if(argc < 2)
		return -1;
	if(argc == 2){
	    cout << "String: " << argv[1] << endl;
	    cout << md5(argv[1]) << endl;
	    return 0;
	}
	if(argc == 3){
		int len = atoi(argv[1]);
		hash_t h = parseHash(argv[2]);
		for(unsigned long long it = 0; it < 1ull<<(6*len); it++){
			char c[len+1];
			c[len]=0;
			genString(c, it, len);
			char r[16];
			memcpy(r,md5(c).c_str(),16);

			int i;
			for(i = 0; i < 16; i++)
				if(r[i] != h.val8[i])
					break;
			if(i == 16){
				std::cout << c << std::endl;
				return 0;
			}
		}
		return 0;
	}
}


