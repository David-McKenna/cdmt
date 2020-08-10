# CUDA PATH
CUDAPATH = $(shell dirname $(shell dirname $(shell which nvcc)))

# Compiling flags
CFLAGS = -I./cuda-samples/Common

CFLAGS_hdf = $(CFLAGS) -I/usr/include/hdf5/serial/
CFLAGS_udp = $(CFLAGS) -I./udpPacketManager/ -Xcompiler "-fopenmp"

# Linking flags
LFLAGS = -lm -L$(CUDAPATH)/lib64 -lcufft -lhdf5 -lcurand
LFLAGS_udp = -lm -L$(CUDAPATH)/lib64 -L./cuda-samples/Common -lcufft -lcurand -Xlinker "-L./udpPacketManager/ -lzstd -llofudpman

# Compilers
NVCC = $(CUDAPATH)/bin/nvcc -arch=sm_70 -O3 --use_fast_math
CC = gcc
CXX = g++

ifeq ($(CC), icc)
LFLAGS_udp += -L$(ONEAPI_ROOT)/compiler/latest/linux/compiler/lib/intel64_lin/ -liomp5 -lirc"
else 
LFLAGS_udp += "
endif

cdmt: git cdmt.o
	$(NVCC) $(CFLAGS_hdf) -o cdmt cdmt.o $(LFLAGS)

cdmt.o: cdmt.cu
	$(NVCC) $(CFLAGS_hdf) -o $@ -c $<

cdmt_udp: git cdmt_udp.o
	$(NVCC) $(CFLAGS_udp) -o cdmt_udp  ./cdmt_udp.o $(LFLAGS_udp)

cdmt_udp.o: cdmt_udp.cu
	$(NVCC) $(CFLAGS_udp) -o $@ -c $<

all: cdmt cdmt_udp

git:
	git submodule update --init --recursive --remote
	cd udpPacketManager; make library CC=$(CC) CXX=$(CXX)

clean:
	rm -f *.o
	rm -f *~
	cd udpPacketManager; make clean
