WARNFLAGS=-Wall -Wextra -pedantic
NVFLAGS= -arch=sm_35 -D_FORCE_INLINES -maxrregcount 96 -gencode arch=compute_35,code=compute_35 -Xptxas -v
#-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
CFLAGS= $(NVFLAGS) -O3 -rdc=true
LDFLAGS=
CC=nvcc

SOURCES=main.cu md5.cu bigInt.cu SHA1.cu utility.cu
OBJECTS=$(SOURCES:.cu=.o)
EXECUTABLE=main

.PHONY:clean
.PHONY:debug

all:$(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

debug:
	$(CC) $(CFLAGS) -g -G -c md5.cu     -o md5.o
	$(CC) $(CFLAGS) -g -G -c main.cu    -o main.o
	$(CC) $(CFLAGS) -g -G -c bigInt.cu  -o bigInt.o
	$(CC) $(CFLAGS) -g -G -c SHA1.cu    -o SHA1.o
	$(CC) $(CFLAGS) -g -G -c utility.cu -o utility
	$(CC) $(LDFLAGS)-g -G $(OBJECTS)    -o $@


clean:
	-rm *.o main debug
