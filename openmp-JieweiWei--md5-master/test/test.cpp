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
#include <chrono>

using std::cout;
using std::endl;

using std::chrono::duration;
using std::chrono::high_resolution_clock;

void printMD5(const string& message) {
  cout << "md5(\"" << message << "\") = "
       << MD5(message).toStr() << endl;
}

std::string getCurrentPwd(int i, char chars[], int len, int n)
{
    std::string result = "";
    for (int j = 0; j < n; j++) {
        result += chars[i % len];
        i /= len;
    }
    return result;
}
int main(int argc, char *argv[]) {
  int n = std::stoi(argv[1]); // password length
  std::string target = argv[2]; // target hash
  std::string result = "";

  cout << "target = " << target << endl;

  high_resolution_clock::time_point start;
  high_resolution_clock::time_point end;
  duration<double, std::milli> duration_sec;
  start = high_resolution_clock::now();

  char chars[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' };
  int len = sizeof(chars) / sizeof(chars[0]);
  
  for (int i = 0; i < (int)pow(len, n); i++) {
      std::string current_pwd = getCurrentPwd(i, chars, len, n);
      if (target == MD5(current_pwd).toStr()) {
          result = current_pwd;
          break;
      }
  }

  end = high_resolution_clock::now();
  duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

  cout << "password is: " << result << endl;
  cout << duration_sec.count() << endl;

	return 0;
}
