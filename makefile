# CUDA PATH
CUDAPATH = /usr/lib/cuda/

# Compiling flags
CFLAGS = -I./cuda-samples/Common

# Linking flags
LFLAGS = -lm -L$(CUDAPATH)/lib64 -lcufft -lhdf5 -lcurand
LFLAGS_udp = -lm -L$(CUDAPATH)/lib64 -L./cuda-samples/Common -lcufft -lcurand

# Compilers
NVCC = $(CUDAPATH)/bin/nvcc
CC = gcc

cdmt: git cdmt.o
	$(NVCC) $(CFLAGS) -o cdmt cdmt.o $(LFLAGS)

cdmt.o: cdmt.cu
	$(NVCC) $(CFLAGS) -o $@ -c $<

cdmt_udp: git cdmt_udp.o
	$(NVCC) $(CFLAGS) -o cdmt_udp cdmt_udp.o $(LFLAGS_udp)

cdmt_udp.o: cdmt_udp.cu
	$(NVCC) $(CFLAGS) -o $@ -c $<

git:
	git submodule update --init --recursive
clean:
	rm -f *.o
	rm -f *~
