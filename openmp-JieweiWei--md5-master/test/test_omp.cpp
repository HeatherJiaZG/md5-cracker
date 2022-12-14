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

char chars[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' };


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

std::string crack(int len, int n, std::string target) {
    std::string result = "";
    for (int i = 0; i < (int)pow(len, n); i++) {
        std::string current_pwd = getCurrentPwd(i, chars, len, n);
        if (target == MD5(current_pwd).toStr()) {
            result = current_pwd;
            break;
        }
    }
    return result;
}

int main(int argc, char *argv[]) {
  int n = std::stoi(argv[1]); // password length
  std::string target = argv[2]; // target hash
  int len = sizeof(chars) / sizeof(chars[0]);
  std::string result = "";

  cout << "target = " << target << endl;

  high_resolution_clock::time_point start;
  high_resolution_clock::time_point end;
  duration<double, std::milli> duration_sec;
  start = high_resolution_clock::now();
  
//   for (int i = 0; i < (int)pow(len, n); i++) {
//       std::string current_pwd = getCurrentPwd(i, chars, len, n);
//       if (target == MD5(current_pwd).toStr()) {
//           result = current_pwd;
//           break;
//       }
//   }

  result = crack(len, n, target);

  end = high_resolution_clock::now();
  duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

  cout << "password is: " << result << endl;
  cout << "Time spent: " << duration_sec.count() << " ms" << endl;

	return 0;
}
