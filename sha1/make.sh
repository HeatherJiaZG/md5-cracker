g++ -O3 -march=native -Wall -Wextra  -std=c++11 -c SHA1.cpp
g++ -O3 -march=native -Wall -Wextra  -std=c++11 -c main.cpp
g++ -O3 -march=native -Wall -Wextra  -std=c++11 SHA1.o main.o -o sha1
