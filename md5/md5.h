/* MD5
 converted to C++ class by Frank Thilo (thilo@unix-ag.org)
 for bzflag (http://www.bzflag.org)

   based on:

   md5.h and md5.c
   reference implementation of RFC 1321

   Copyright (C) 1991-2, RSA Data Security, Inc. Created 1991. All
rights reserved.

License to copy and use this software is granted provided that it
is identified as the "RSA Data Security, Inc. MD5 Message-Digest
Algorithm" in all material mentioning or referencing this software
or this function.

License is also granted to make and use derivative works provided
that such works are identified as "derived from the RSA Data
Security, Inc. MD5 Message-Digest Algorithm" in all material
mentioning or referencing the derived work.

RSA Data Security, Inc. makes no representations concerning either
the merchantability of this software or the suitability of this
software for any particular purpose. It is provided "as is"
without express or implied warranty of any kind.

These notices must be retained in any copies of any part of this
documentation and/or software.

*/

#ifndef BZF_MD5_H
#define BZF_MD5_H

#include <cstring>
#include <iostream>


// a small class for calculating MD5 hashes of strings or byte arrays
// it is not meant to be fast or secure
//
// usage: 1) feed it blocks of uchars with update()
//      2) finalize()
//      3) get hexdigest() string
//      or
//      MD5(std::string).hexdigest()
//
// assumes that char is 8 bit and int is 32 bit
class MD5{
public:

  MD5();
  MD5(const std::string& text);
  void update(const unsigned char *buf, unsigned length);
  void update(const char *buf, unsigned length);
  MD5& finalize();
  std::string hexdigest() const;
  friend std::ostream& operator<<(std::ostream&, MD5 md5);

private:
  void init();
  const static unsigned char blocksize = 64; // VC6 won't eat a const static int here

  void transform(const unsigned char block[blocksize]);
  static void decode(unsigned output[], const unsigned char input[], unsigned len);
  static void encode(unsigned char output[], const unsigned input[], unsigned len);

  bool finalized;
  unsigned char buffer[blocksize]; // bytes that didn't fit in last 64 byte chunk
  unsigned count[2];   // 64bit counter for number of bits (lo, hi)
  unsigned state[4];   // digest so far
  unsigned char digest[16]; // the result

  // low level logic operations
  static inline unsigned F(unsigned x, unsigned y, unsigned z);
  static inline unsigned G(unsigned x, unsigned y, unsigned z);
  static inline unsigned H(unsigned x, unsigned y, unsigned z);
  static inline unsigned I(unsigned x, unsigned y, unsigned z);
  static inline unsigned rotate_left(unsigned x, int n);
  static inline void FF(unsigned &a, unsigned b, unsigned c, unsigned d, unsigned x, unsigned s, unsigned ac);
  static inline void GG(unsigned &a, unsigned b, unsigned c, unsigned d, unsigned x, unsigned s, unsigned ac);
  static inline void HH(unsigned &a, unsigned b, unsigned c, unsigned d, unsigned x, unsigned s, unsigned ac);
  static inline void II(unsigned &a, unsigned b, unsigned c, unsigned d, unsigned x, unsigned s, unsigned ac);
};

std::string md5(const std::string str);

#endif