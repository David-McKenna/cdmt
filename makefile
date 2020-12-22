# CUDA PATH
CUDAPATH = $(shell dirname $(shell dirname $(shell which nvcc)))

# Compiling flags
CFLAGS = -I./cuda-samples/Common

CFLAGS_hdf = $(CFLAGS) -I/usr/include/hdf5/serial/
CFLAGS_udp = $(CFLAGS) -Xcompiler "-fopenmp"

# Linking flags
LFLAGS = -lm -L$(CUDAPATH)/lib64 -lcufft -lhdf5 -lcurand
LFLAGS_udp = -lm -L$(CUDAPATH)/lib64 -L./cuda-samples/Common -lcufft -lcurand -Xlinker "-lzstd -llofudpman


# Compilers
CC = gcc
CXX = g++
NVCC = $(CUDAPATH)/bin/nvcc -arch=sm_70 -O3 --use_fast_math -ccbin=$(CXX)

ifeq ($(CC), icc)
LFLAGS_udp += -L$(ONEAPI_ROOT)/compiler/latest/linux/compiler/lib/intel64_lin/ -liomp5 -lirc"
else 
LFLAGS_udp += -lgomp"
endif

cdmt: git cdmt.o
	$(NVCC) $(CFLAGS_hdf) -o cdmt cdmt.o $(LFLAGS)

cdmt.o: cdmt.cu
	$(NVCC) $(CFLAGS_hdf) -o $@ -c $<

cdmt_udp: git cdmt_udp.o
	$(NVCC) $(CFLAGS_udp) -o cdmt_udp  ./cdmt_udp.o $(LFLAGS_udp)

cdmt_udp.o: cdmt_udp.cu
	$(NVCC) $(CFLAGS_udp) -o $@ -c $<

cdmt_udp_stokesV: stokesVPrep cdmt_udp_stokesV.o
	$(NVCC) $(CFLAGS_udp) -o cdmt_udp_stokesV  ./cdmt_udp_stokesV.o $(LFLAGS_udp)

stokesVPrep:
	cp ./cdmt_udp.cu ./cdmt_udp_stokesV.cu
	sed -i 's/cp1\[idx1\]\.x\*cp1\[idx1\]\.x+cp1\[idx1\]\.y\*cp1\[idx1\]\.y+cp2\[idx1\]\.x\*cp2\[idx1\]\.x+cp2\[idx1\]\.y\*cp2\[idx1\]\.y/2.0 \* ((cp1\[idx1\]\.x \* cp2\[idx1\]\.y) - (cp1\[idx1\]\.y \* cp2\[idx1\]\.x))/g' cdmt_udp_stokesV.cu

cdmt_udp_stokesV.o:cdmt_udp_stokesV.cu
	$(NVCC) $(CFLAGS_udp) -o $@ -c $<

all: cdmt cdmt_udp cdmt_udp_stokesV

git:
	git submodule update --init --recursive --remote

clean:
	rm -f *.o
	rm -f *~
	rm *stokesV.cu; exit 0;

