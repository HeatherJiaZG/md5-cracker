#include <cuda.h>
#include <stdint.h>
#include <string.h>
#include <string>
#include <bits/stdc++.h>
#include <iostream>
#include "md5.cu"

using namespace std;
 

std::string getCurrentPwd(int i, char chars[], int len, int n)
{
    std::string result = "";
    for (int j = 0; j < n; j++) {
        result += chars[i % len];
        i /= len;
    }
    return result;
}

char* hash(const char* h_str) {
    char* d_str;
    unsigned char* h_res = (unsigned char*)malloc(sizeof(unsigned char)*(32 + 1));
    unsigned char* d_res;
    cudaMalloc((void**)&d_str, sizeof(char) * strlen(h_str));
    cudaMalloc((void**)&d_res, sizeof(char) * 32);
    cudaMemcpy(d_str, h_str, sizeof(char) * strlen(h_str), cudaMemcpyHostToDevice);

    md5<<<1, 1>>>(d_str, (uint32_t)strlen(h_str), d_res);

    cudaMemcpy(h_res, d_res, sizeof(unsigned char)*(32), cudaMemcpyDeviceToHost);

    cudaFree(d_str);
    cudaFree(d_res);

    char* res = (char*)malloc(sizeof(char)*32);
    for (int i = 0; i < 16; i++) {
        sprintf(&res[i*2], "%2.2x", h_res[i]);
    }
    return res;
}

int run_test(const char* name, const char* result, const char* expected) {
    if (strcmp(expected, result) == 0) {
        printf("TEST PASSED: %s: expected %s, got %s\n", name, expected, result);
        return 1;
    } else {
        printf("TEST FAILED: %s: expected %s, got %s\n", name, expected, result);
        return 0;
    }
}


int main() {
    // int passed = 0, failed = 0;

    // run_test("md5(\"\")", hash(""), "d41d8cd98f00b204e9800998ecf8427e") ? passed++ : failed++;
    // run_test("md5(\"a\")", hash("a"), "0cc175b9c0f1b6a831c399e269772661") ? passed++ : failed++;
    // run_test("md5(\"abc\")", hash("abc"), "900150983cd24fb0d6963f7d28e17f72") ? passed++ : failed++;

    // run_test("md5(\"bad\")", hash("bad"), "bae60998ffe4923b131e3d6e4c19993e") ? passed++ : failed++;
    // run_test("md5(\"message digest\")", hash("message digest"), "f96b697d7cb7938d525a2f31aaf161d0") ? passed++ : failed++;
    // run_test("md5(\"abcdefghijklmnopqrstuvwxyz\")", \
    //     hash("abcdefghijklmnopqrstuvwxyz"), \
    //     "c3fcd3d76192e4007dfb496cca67e13b") ? passed++ : failed++;
    // run_test("md5(\"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789\")", \
    //     hash("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"), \
    //     "d174ab98d277d9f5a5611c2c9f419d9f") ? passed++ : failed++;
    // run_test("md5(\"12345678901234567890123456789012345678901234567890123456789012345678901234567890\")", \
    //     hash("12345678901234567890123456789012345678901234567890123456789012345678901234567890"), \
    //     "57edf4a22be3c955ac49da2e2107b67a") ? passed++ : failed++;

    // printf("Tests Passed: %i\n", passed);
    // printf("Tests Failed: %i\n", failed);
    // return failed;


    // int n = stoi(argv[1]); // password length
    int n=3;
    // std::string target = argv[2]; // target hash
    std::string target = "bae60998ffe4923b131e3d6e4c19993e";
    std::string result = "";

    cout << "target = " << target << endl;

    char chars[] = { 'a', 'b', 'd', '3', '4', '5', '6', '7', '8', '9' };
    int len = sizeof(chars) / sizeof(chars[0]);
    for (int i = 0; i < (int)pow(len, n); i++) {
        std::string current_pwd = getCurrentPwd(i, chars, len, n);
        if(strcmp(target, hash(current_pwd)) == 0) {
            result = current_pwd;
            break;
        }
    }
    printf("password is %s\n", result);

}