# CUDA PATH
CUDAPATH = /usr/$(shell test -d /usr/lib/cuda && echo lib/cuda || echo local/cuda)

# Compiling flags
CFLAGS = -std=c++11 -I./cuda-samples/Common -I/usr/include/hdf5/serial/ -I./udpPacketManager/ -Xcompiler -fopenmp

# Linking flags
LFLAGS = -lm -L$(CUDAPATH)/lib64 -lcufft -lhdf5 -lcurand
LFLAGS_udp = -lm -L$(CUDAPATH)/lib64 -L./cuda-samples/Common -lcufft -lcurand -lzstd -lgomp

# Compilers
NVCC = $(CUDAPATH)/bin/nvcc -arch=sm_70 -O3 --use_fast_math
CC = gcc

cdmt: git cdmt.o
	$(NVCC) $(CFLAGS) -o cdmt cdmt.o $(LFLAGS)

cdmt.o: cdmt.cu
	$(NVCC) $(CFLAGS) -o $@ -c $<

cdmt_udp: git cdmt_udp.o
	$(NVCC) $(CFLAGS) -o cdmt_udp  ./cdmt_udp.o ./udpPacketManager/lofar_udp_reader.o ./udpPacketManager/lofar_udp_misc.o $(LFLAGS_udp) -Xcompiler -fopenmp,-L./udpPacketManager/

cdmt_udp.o: cdmt_udp.cu
	$(NVCC) $(CFLAGS) -o $@ -c $<

git:
	git submodule update --init --recursive --remote
	cd udpPacketManager; make all
clean:
	rm -f *.o
	rm -f *~
	cd udpPacketManager; make clean
