/**
 * @file test.cpp
 * @The test file of md5.
 * @author Jiewei Wei
 * @mail weijieweijerry@163.com
 * @github https://github.com/JieweiWei
 * @data Oct 19 2014
 *
 */

#include <iostream>
#include "../src/md5.h"
#include <string>
#include <bits/stdc++.h>

using std::cout;
using std::endl;

void printMD5(const string& message) {
  cout << "md5(\"" << message << "\") = "
       << MD5(message).toStr() << endl;
}

int main() {
  int n = stoi(argv[1]); // password length
  std::string target = argv[2]; // target hash
  std::string result = "";

  cout << "target = " << target << endl;

  char chars[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' };
  int len = sizeof(chars) / sizeof(chars[0]);
  for (int i = 0; i < (int)pow(len, n); i++) {
      std::string current_pwd = getCurrentPwd(i, chars, len, n);
      if (target == MD5(current_pwd).toStr()) {
          result = current_pwd;
          break;
      }
  }
  cout << "password is: " << result << endl;
	return 0;
}
