# CUDA PATH
CUDAPATH = /usr/lib/cuda/

export OMP_CANCELLATION=1

# Compiling flags
CFLAGS = -g -I./cuda-samples/Common -I/usr/include/hdf5/serial/ -I./udpPacketManager/

# Linking flags
LFLAGS = -lm -L$(CUDAPATH)/lib64 -lcufft -lhdf5 -lcurand
LFLAGS_udp = -lm -L$(CUDAPATH)/lib64 -L./cuda-samples/Common -lcufft -lcurand -lzstd


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

cdmt_udp_stream: git cdmt_udp_stream.o
	$(NVCC) $(CFLAGS) -o cdmt_udp_stream  ./cdmt_udp_stream.o ./udpPacketManager/lofar_udp_reader.o ./udpPacketManager/lofar_udp_misc.o $(LFLAGS_udp) -Xcompiler -fopenmp,-L./udpPacketManager/

cdmt_udp_stream.o: cdmt_udp_stream.cu
	$(NVCC) $(CFLAGS) -o $@ -c $< -Xcompiler -fopenmp

git:
	git submodule update --init --recursive
clean:
	rm -f *.o
	rm -f *~
