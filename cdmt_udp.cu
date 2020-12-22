#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <cuda.h>
#include <cufft.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <getopt.h>
#include <limits.h>
#include <omp.h>

#include "lofar_udp_reader.h"
#include "lofar_udp_misc.h"

// Timing macro
#ifndef __LOFAR_UDP_TICKTOCK_MACRO
#define __LOFAR_UDP_TICKTOCK_MACRO
// XOPEN -> strptime requirement
#define __USE_XOPEN
#include <time.h>

#define CLICK(clock) clock_gettime(CLOCK_MONOTONIC_RAW, &clock);
#define TICKTOCK(tick, tock) ((double) (tock.tv_nsec - tick.tv_nsec) / 1000000000.0) + (tock.tv_sec - tick.tv_sec)
#endif


#define HEADERSIZE 4096
#define DMCONSTANT 2.41e-10
#define VERB 1

// Struct for header information
struct header {
  int nchan,nbit=0,nsub,tel=11,mach=11;
  double tstart,tsamp,fch1,foff,fcen,bwchan;
  double src_raj,src_dej;
  char source_name[80];
  char rawfname[4][1024];
};


// Prototypes
struct header read_sigproc_header(char *fname, char *dataname, int ports);
void get_channel_chirp(double fcen,double bw,float dm,int nchan,int nbin,int nsub,cufftComplex *c);
__global__ void transpose_unpadd_and_detect(cufftComplex *cp1,cufftComplex *cp2,int nbin,int nchan,int nfft,int nsub,int noverlap,int nsamp,float *fbuf);
static __device__ __host__ inline cufftComplex ComplexScale(cufftComplex a,float s);
static __device__ __host__ inline cufftComplex ComplexMul(cufftComplex a,cufftComplex b);
static __global__ void PointwiseComplexMultiply(cufftComplex *a,cufftComplex *b,cufftComplex *c,int nx,int ny,int l,float scale);
template<typename I> __global__ void unpack_and_padd(I *dbuf0,I *dbuf1,I *dbuf2,I *dbuf3,int nsamp,int nbin,int nfft,int nsub,int noverlap,cufftComplex *cp1,cufftComplex *cp2);
template<typename I> __global__ void unpack_and_padd_first_iteration(I *dbuf0,I *dbuf1,I *dbuf2,I *dbuf3,int nsamp,int nbin,int nfft,int nsub,int noverlap,cufftComplex *cp1,cufftComplex *cp2);
template<typename I> __global__ void padd_next_iteration(I *dbuf0,I *dbuf1,I *dbuf2,I *dbuf3,int nsamp,int nbin,int nfft,int nsub,int noverlap,cufftComplex *cp1,cufftComplex *cp2);
__global__ void swap_spectrum_halves(cufftComplex *cp1,cufftComplex *cp2,int nx,int ny);
__global__ void compute_chirp(double fcen,double bw,float *dm,int nchan,int nbin,int nsub,int ndm,cufftComplex *c);
__global__ void compute_block_sums(float *z,int nchan,int nblock,int nsum,float *bs1,float *bs2);
__global__ void compute_channel_statistics(int nchan,int nblock,int nsum,float *bs1,float *bs2,float *zavg,float *zstd);
__global__ void redigitize(float *z,int nchan,int nblock,int nsum,float *zavg,float *zstd,float zmin,float zmax,unsigned char *cz);
__global__ void decimate_and_redigitize(float *z,int ndec,int nchan,int nblock,int nsum,float *zavg,float *zstd,float zmin,float zmax,unsigned char *cz);
__global__ void decimate(float *z,int ndec,int nchan,int nblock,int nsum,float *cz);
void write_to_disk_float(float* outputArray, FILE** outputFile, int nsamples, cudaEvent_t* waitEvent);
void write_to_disk_char(unsigned char* outputArray, FILE** outputFile, int nsamples, cudaEvent_t* waitEvent);
void write_filterbank_header(struct header h,FILE *file);
int reshapeRawUdp(lofar_udp_reader *reader, int verbose);
long  __inline__ beamformed_packno(unsigned int timestamp, unsigned int sequence);
long getStartingPacket(char inputTime[], const int clock200MHz);


long getStartingPacket(char inputTime[], const int clock200MHz) {
  struct tm unixTm;
  time_t unixEpoch;

  if(strptime(inputTime, "%Y-%m-%dT%H:%M:%S", &unixTm) != NULL) {
    unixEpoch = timegm(&unixTm);
    return beamformed_packno((unsigned long int) unixEpoch, 0, clock200MHz);
  } else {
    fprintf(stderr, "Invalid time string, exiting.\n");
    return 1;
  }

}
// External prototypes from udpPacketManager
extern "C"
{
  int lofar_udp_reader_step(lofar_udp_reader *reader);
  lofar_udp_reader* lofar_udp_meta_file_reader_setup(FILE **inputFiles, const int numPorts, const int replayDroppedPackets, const int processingMode, const int verbose, const long packetsPerIteration, const long startingPacket, const long packetsReadMax, const int compressedReader);
  int lofar_udp_file_reader_reuse(lofar_udp_reader *reader, const long startingPacket, const long packetsReadMax);
}

// Usage
void usage()
{
  printf("cdmt -v -c -d <DM start,step,num> -D <GPU device> -b <ndec> -N <forward FFT size> -n <overlap region> -f <number of FFTs per operation> -o <outputname> -s <sigproc header location> -p <port nums> <fil prefix>\n\n");
  printf("Compute coherently dedispersed SIGPROC filterbank files from LOFAR complex voltage data in raw udp format.\n");
  printf("-D <GPU device>  Select GPU device [integer, default: 0]\n");
  printf("-b <ndec>        Number of time samples to average [integer, default: 1]\n");
  printf("-d <DM start, step, num>  DM start and stepsize, number of DM trials\n");
  printf("-o <outputname>           Output filename [default: cdmt]\n");
  printf("-N <forward FFT size>     Forward FFT size [integer, default: 65536]\n");
  printf("-n <overlap region>       Overlap region [integer, default: 2048]\n");
  printf("-s <ISOT str>    Time to skip to when starting to process data [default: "", only supports 200MHz clock]\n");
  printf("-r <packets>     Number of packets to read in total from the -s offset [integer, default: length of file]\n");
  printf("-m <hdr loc>     Sigproc header to read metadata from [default: fil prefix.sigprochdr]\n");
  printf("-f <FFTs per op> Number of FFTs to execute per cuFFT call [default: 128]\n");
  printf("-a               Disable redigitisation; output float32 [default: false]\n");
  printf("-c <num chan>    Channelisation Factor [default: 8]\n");
  printf("-w               Print warnings about input parameter sizes and packet loss [default: true]\n");
  printf("-p <num>         Number of ports of data to process [default: 4]\n");
  printf("-l <num>         Base port number to iterate from when determining raw file names [default for IE613: 16130]\n");
  printf("-t               Perform a dry run; proceed as expected until we would start processing data.\n");
  printf("-z <subband strategy>     Apply dreamBeam corrections to voltages (default: false).\n");
  printf("-Z <lower>,<upper>        Extract only the specific beamlets (default: all)\n");

  return;
}

int main(int argc,char *argv[])
{
  int i,j,nsamp,nfft,mbin,nvalid,nchan=8,nbin=65536,noverlap=2048,nsub=20,ndm,ndec=1;
  int idm,nread_tmp,nread,mchan,msamp,mblock,msum=1024;
  char *header,*udpbuf[4],*dudpbuf_c[4];
  FILE *file;
  unsigned char **cbuf[2],*dcbuf;
  float **cbuff[2], *dcbuff, *dudpbuf_f[4];
  float *fbuf,*dfbuf;
  float *bs1,*bs2,*zavg,*zstd;
  cufftComplex *cp1,*cp2,*dc,*cp1p,*cp2p;
  cufftHandle ftc2cf,ftc2cb;
  int idist,odist,iembed,oembed,istride,ostride;
  dim3 blocksize,gridsize;
  struct header hdr;
  float *dm,*ddm,dm_start,dm_step;
  char fname[128],fheader[1024],*udpfname,sphdrfname[1024] = "",obsid[128]="cdmt",inputTime[128]="",subbands[4096]="";
  int bytes_read;
  long int ts_read=LONG_MAX;
  long int total_ts_read=0;
  int part=0,device=0,nforward=128,redig=1,ports=4,baseport=16130,checkinputs=1,testmode=0,dreamBeam=0,beamletLower=0,beamletUpper=0;
  int arg=0;
  FILE **outfile;
  struct timespec tick, tick0, tick1, tock, tock0;
  double elapsedTime;

  lofar_udp_reader *reader;

  // Read options
  if (argc>1) {
    while ((arg=getopt(argc,argv,"tawc:p:f:d:D:ho:b:N:n:s:r:m:t:p:l:z:Z:"))!=-1) {
      switch (arg) {
  
      case 'n':
  noverlap=atoi(optarg);
  break;

      case 'N':
  nbin=atoi(optarg);
  break;

      case 'b':
  ndec=atoi(optarg);
  break;

      case 'o':
  strcpy(obsid,optarg);
  break;

      case 'D':
  device=atoi(optarg);
  break;

      case 's':
  strcpy(inputTime, optarg);
  break;
  
      case 'r':
  ts_read=atol(optarg);
  break;

      case 'd':
  sscanf(optarg,"%f,%f,%d",&dm_start,&dm_step,&ndm);
  break;

      case 'm':
  strcpy(sphdrfname,optarg);
  break;

      case 'f':
  nforward=atoi(optarg);
  break;
  
      case 'a':
  redig=0;
  break;

      case 'c':
  nchan=atoi(optarg);
  break;

      case 'w':
  checkinputs=0;
  break;

      case 'p':
  ports=atoi(optarg);
  break;

      case 'l':
  baseport=atoi(optarg);
  break;

      case 't':
  testmode=1;
  break;

      case 'z':
  dreamBeam = 1;
  strcpy(subbands, optarg);
  break;

      case 'Z':
  sscanf(optarg, "%d,%d", &beamletLower,&beamletUpper);
  break;

      case 'h':
  usage();
  return 0;

      }
    }
  } else {
    printf("Unknown option '%c'\n", arg);
    usage();
    return 0;
  }

  if (argc <= optind) {
    fprintf(stderr, "Failed to provide a source file, exiting.\n");
    return 0;
  }
  udpfname=argv[optind];

  // Sanity checks to avoid voids in output filterbank
  if (checkinputs) 
  {
    if (nbin % nchan != 0) {
      fprintf(stderr, "ERROR: nbin must be disible by nchan (%d) (currently %d, remainder %d). Exiting.\n", nchan, nbin, nbin % nchan);
      exit(1);
    }
    if ((nforward * (nbin-2*noverlap)) % nchan != 0 ) {
      fprintf(stderr, "ERROR: Valid data length must be divisible by nchan (%d) (currently %d, remainer %d). Exiting.\n", nchan, nbin-2*noverlap, (nbin-2*noverlap) % nchan);
      exit(1);
    }

    if ((nforward * (nbin-2*noverlap) / nchan) % msum != 0) {
      printf("%d\n", (nforward * (nbin-2*noverlap) / nchan) % msum);
      fprintf(stderr, "ERROR: Interal sum cannot proceed; valid samples must be divisible by msum (%d) (currently %d, remainder %d).\n", msum, (nforward * (nbin-2*noverlap) / nchan), (nforward * (nbin-2*noverlap) / nchan) % msum);
      exit(1);
    }
    
    if ((nforward * (nbin-2*noverlap)) % 16 != 0) {
      fprintf(stderr, "ERROR: Number of valid samples must be divisible by samples per packet (16) (currently %d, remainder %d). Exiting.\n", (nforward * (nbin-2*noverlap)), (nforward * (nbin-2*noverlap)) % 16);
      exit(1);
    }
  }

  long startingPacket;
  if (strcmp(inputTime, "") != 0) {
    startingPacket = getStartingPacket(inputTime, 1);
    printf("Skipping to packet %ld (%s)\n", startingPacket, inputTime);
  } else{
    startingPacket = -1;
  }


  // Error if given an invalid sigproc header location
  if (strcmp(sphdrfname, "") == 0) {
    fprintf(stderr, "ERROR: Sigproc header not provided. Exiting.\n");
    exit(1);
  }

  FILE* inputFiles[4];
  int compressedInput = 0;
  char tmpfname[1024] = "";
  // Pattern check to determine if files are compressed or not
  if (strstr(udpfname, ".zst") != NULL) compressedInput = 1;


 
  // Read the provided sigproc header
  hdr = read_sigproc_header(sphdrfname, udpfname, ports);

  // Update to account for runtime-dropped dropped beamlets
  if (beamletUpper != 0) {
    hdr.fch1 += hdr.foff * (nsub - beamletUpper);
    nsub = beamletUpper;
  }

  if (beamletLower != 0) {
    nsub -= beamletLower;
  }



  // Check that the bin size and overlap sizes are sufficiently large to avoid issues with the convolution theorem
  if (checkinputs)
  {
    const double stg1 = (1.0 / 2.41e-4) *  abs(pow((double) hdr.fch1 + hdr.nsub * hdr.foff + hdr.foff *0.5,-2.0) - pow((double) hdr.fch1 + hdr.nsub * hdr.foff - hdr.foff *0.5, -2.0)) * (dm_start + dm_step * (ndm - 1));
    const int overlapCheck = (int) (stg1 / hdr.tsamp);
    if (overlapCheck > nbin) {
      fprintf(stderr, "WARNING: The size of your FFT bin is too short for the given DMs and frequencies. Given bin size: %d, Suggested minimum bin size: %d (maximum dispersion delay %f).\n", nbin, overlapCheck, stg1);
    } else if (overlapCheck / 2 > noverlap) {
      fprintf(stderr, "WARNING: The size of your FFT overlap is too short for the given maximum DM. Given overlap: %d, Suggested minimum overlap: %d (maximum dispersion delay %f).\n", noverlap, overlapCheck / 2, stg1);
    }
  }


  // Open raw files
  for (int i = 0; i < ports; i++) {
    sprintf(tmpfname, udpfname, i + baseport);
    if (strcmp(udpfname, tmpfname) == 0 && ports > 1) {
      fprintf(stderr, "ERROR: Input file name has not changed when attempting to substitute in port, have you correctly defined your file name?\n");
      exit(1);
    }
    printf("Opening %s...\n", tmpfname);

    inputFiles[i] = fopen(tmpfname, "r");

    if (inputFiles[i] == NULL) {
      printf("Input file failed to open (null pointer)\n");
    }

    if (i == 0)
      strcpy(hdr.rawfname[i],tmpfname);
  }

  // Read the number of raw subbands + sampling time for later
  nsub=hdr.nsub;
  double timeOffset = hdr.tsamp;

  // Adjust header for the target output
  hdr.tsamp*=nchan*ndec;
  hdr.nchan=nsub*nchan;
  if (redig) hdr.nbit=8;
  else hdr.nbit=32;
  hdr.fch1=hdr.fcen+0.5*hdr.nsub*hdr.bwchan-0.5*hdr.bwchan/nchan;
  hdr.foff=-fabs(hdr.bwchan/nchan);


  // Data sizes; these control the block sizes for interall logic components
  // FFTs are performed on a 3D block; of dims (nfft, nsub, nbin), there are
  // overlaps of noverlap samples between each nbin element
  nvalid=nbin-2*noverlap;
  nsamp=nforward*nvalid;
  nfft=(int) ceil(nsamp/(float) nvalid);
  mbin=nbin/nchan; // nbin must be evenly divisible by nchan
  mchan=nsub*nchan;
  msamp=nsamp/nchan; // nforward * nvalid must be divisble by nchan
  mblock=msamp/msum; // nforward * nvalid / nchan must be disible by msum

  // Determine the number of packets we need to request per iteration
  const long int packetGulp = nsamp / 16;

  printf("Configuring reader...\n");
  lofar_udp_config udp_cfg = lofar_udp_config_default;

  udp_cfg.inputFiles = inputFiles;
  udp_cfg.numPorts = ports;
  udp_cfg.replayDroppedPackets = 1;
  udp_cfg.processingMode = 11;
  udp_cfg.verbose = 0;
  udp_cfg.packetsPerIteration = packetGulp;
  udp_cfg.startingPacket = startingPacket;
  udp_cfg.packetsReadMax = LONG_MAX;
  udp_cfg.compressedReader = compressedInput;
  udp_cfg.beamletLimits[0] = beamletLower;
  udp_cfg.beamletLimits[1] = beamletUpper;
  udp_cfg.calibrateData = dreamBeam;

  lofar_udp_calibration udp_cal = lofar_udp_calibration_default;
  udp_cfg.calibrationConfiguration = &udp_cal;
  char fifo[128] = "/tmp/dreamBeamCDMTFIFO", rajs[64], decjs[64];
  float raj = 0.0, decj = 0.0, ras, decs;
  int rah, ram, decd, decm;
  if (dreamBeam == 1) {
    sprintf(rajs, "%012.5f", hdr.src_raj);
    sprintf(decjs, "%012.5f", hdr.src_dej);

    printf("rajs %s, decjs %s\n", rajs, decjs);
    sscanf(rajs, "%02d%02d%f", &rah, &ram, &ras);
    sscanf(decjs, "%02d%02d%f", &decd, &decm, &decs);

    raj = (float) rah * 0.26179958333 + (float) ram * 0.00436332638 + ras * 0.0000727221;
    decj = (float) abs(decd) * 0.0174533 + (float) decm * 0.00029088833 + decs * 0.00000484813;

    if (hdr.src_dej < 0)
      decj *= -1;

    printf("Raj: %f, Decj: %f\n", raj, decj);
    printf("Configuring calibration...\n");
    strcpy(udp_cal.calibrationFifo, fifo);
    strcpy(udp_cal.calibrationSubbands, subbands);
    udp_cal.calibrationPointing[0] = raj;
    udp_cal.calibrationPointing[1] = decj;
    strcpy(udp_cal.calibrationPointingBasis, "J2000");
  }


  printf("Loading %ld packets per gulp. Setting up reader...\n", packetGulp);
  reader = lofar_udp_meta_file_reader_setup_struct(&udp_cfg);

  printf("Reader: %d, %d.\t %d, %d.\t %d, %ld, %d\n", reader->meta->totalRawBeamlets, reader->meta->totalProcBeamlets, reader->meta->inputBitMode, reader->meta->outputBitMode, reader->meta->processingMode, reader->meta->packetsPerIteration, reader->meta->replayDroppedPackets);

  if (reader == NULL) {
    fprintf(stderr, "Failed to generate LOFAR UDP Reader, exiting.\n");
    exit(1);
  }

  if (reader->meta->totalProcBeamlets != nsub) {
    fprintf(stderr, "ERROR: Number of beamlets does not match number of channels in header, exiting.\n");
    exit(1);
  }

  // Update the start time based on the first provided packet
  hdr.tstart = lofar_get_packet_time_mjd(reader->meta->inputData[0]);

  // Set device
  checkCudaErrors(cudaSetDevice(device));

  // Generate streams for asyncrnous operations
  int numStreams = 1;
  // Add an extra stream for the final padding operation
  cudaStream_t streams[numStreams+1];
  for (i = 0; i < numStreams+1; i++)
    checkCudaErrors(cudaStreamCreate(&(streams[i])));

  // Create 2 events; one which blocks execution (preventing new data reads) and the other waiting for compute to finish.
  int numEvents = 3;
  cudaEvent_t events[numEvents];
  cudaEventCreateWithFlags(&(events[0]), cudaEventBlockingSync & cudaEventDisableTiming);
  cudaEventCreateWithFlags(&(events[1]), cudaEventDisableTiming);
  cudaEventCreateWithFlags(&(events[2]), cudaEventDisableTiming);

  cudaEvent_t dmWriteEvents[numStreams][ndm];
  for (i =0; i < ndm; i++)
    for (j = 0; j < numStreams; j++)
      cudaEventCreateWithFlags(&(dmWriteEvents[j][i]), cudaEventBlockingSync & cudaEventDisableTiming);

  // DMcK: cuFFT docs say it's best practice to plan before allocating memory
  // cuda-memcheck fails initialisation before this block is run? -- add CUDA_MEMCHECK_PATCH_MODULE=1 as an env flag

  // Disable initial memory allocates and silence the compiler  warnings; 
  // Nvidia uses a custom compiler frontend so GCC pragmas do not work.
  // This order-of-execution follows Nvidia's usage guidance, so the warnings
  // should be (safely?) ignored
  #pragma diag_suppress used_before_set
  cufftSetAutoAllocation(ftc2cf, 0);
  cufftSetAutoAllocation(ftc2cb, 0);
  #pragma diag_default used_before_set
  size_t cfSize, cbSize;

  // Generate FFT plan (batch in-place forward FFT)
  idist=nbin;  odist=nbin;  iembed=nbin;  oembed=nbin;  istride=1;  ostride=1;
  checkCudaErrors(cufftPlanMany(&ftc2cf,1,&nbin,&iembed,istride,idist,&oembed,ostride,odist,CUFFT_C2C,nfft*nsub));
  checkCudaErrors(cufftGetSizeMany(ftc2cf, 1,&nbin,&iembed,istride,idist,&oembed,ostride,odist,CUFFT_C2C,nfft*nsub, &cfSize));
  //cufftSetStream(ftc2cf,streams[0]);

  // Generate FFT plan (batch in-place backward FFT)
  idist=mbin;  odist=mbin;  iembed=mbin;  oembed=mbin;  istride=1;  ostride=1;
  checkCudaErrors(cufftPlanMany(&ftc2cb,1,&mbin,&iembed,istride,idist,&oembed,ostride,odist,CUFFT_C2C,nchan*nfft*nsub));
  checkCudaErrors(cufftGetSizeMany(ftc2cb, 1,&mbin,&iembed,istride,idist,&oembed,ostride,odist,CUFFT_C2C,nchan*nfft*nsub,&cbSize));
  //cufftSetStream(ftc2cb,streams[0]);
  cudaDeviceSynchronize();

  // Get the maximum size needed for the FFT operations (they should be the same, check for safety)
  size_t minfftSize = cfSize > cbSize ? cfSize : cbSize;
  if (VERB) printf("Allocated %ldMB for cuFFT work (saving %ldMB)\n", minfftSize >> 20, (cfSize + cbSize - minfftSize) >> 20);

  // Predict the overall VRAM usage
  long unsigned int bytesUsed = sizeof(cufftComplex) * nbin * nfft * nsub * 4 + sizeof(cufftComplex) * nbin *nsub * ndm + sizeof(float) * mblock * mchan * 2 + (sizeof(char) * (1 - dreamBeam) + sizeof(float) * dreamBeam) * nsamp * nsub * 4 + sizeof(float) * nsamp * nsub + redig * msamp * mchan / ndec - (redig - 1) * 4 * msamp * mchan * ndec;
  
  // Get the total / available VRAM
  size_t gpuMems[2];
  checkCudaErrors(cudaMemGetInfo(&(gpuMems[0]), &(gpuMems[1])));
  if (VERB) printf("Preparing for GPU memory allocations. Current memory usage: %ld / %ld GB\n", (gpuMems[1] - gpuMems[0]) >> 30, gpuMems[1] >> 30);
  if (VERB) printf("We anticipate %ld MB (%ld GB) to be allocated on the GPU (%ld MB for cuFFT planning).\n", (bytesUsed + minfftSize) >> 20, (bytesUsed + minfftSize) >> 30, minfftSize >> 20);


  // Allocate the maxmimum memory needed for FFT operations
  void* cufftWorkArea;
  checkCudaErrors(cudaMalloc((void**) &cufftWorkArea, (size_t) minfftSize));
  // Set the cuFFT handles to use this area
  cufftSetWorkArea(ftc2cf, cufftWorkArea);
  cufftSetWorkArea(ftc2cb, cufftWorkArea);


  // Allocate memory for complex timeseries
  checkCudaErrors(cudaMalloc((void **) &cp1,  (size_t) sizeof(cufftComplex)*nbin*nfft*nsub));
  checkCudaErrors(cudaMalloc((void **) &cp2,  (size_t) sizeof(cufftComplex)*nbin*nfft*nsub));
  checkCudaErrors(cudaMalloc((void **) &cp1p, (size_t) sizeof(cufftComplex)*nbin*nfft*nsub));
  checkCudaErrors(cudaMalloc((void **) &cp2p, (size_t) sizeof(cufftComplex)*nbin*nfft*nsub));

  // Allocate device memory for chirp
  checkCudaErrors(cudaMalloc((void **) &dc, (size_t) sizeof(cufftComplex)*nbin*nsub*ndm));

  if (redig) {
    // Allocate device memory for block sums
    checkCudaErrors(cudaMalloc((void **) &bs1, (size_t) sizeof(float)*mblock*mchan));
    checkCudaErrors(cudaMalloc((void **) &bs2, (size_t) sizeof(float)*mblock*mchan));

    // Allocate device memory for channel averages and standard deviations
    checkCudaErrors(cudaMalloc((void **) &zavg, (size_t) sizeof(float)*mchan));
    checkCudaErrors(cudaMalloc((void **) &zstd, (size_t) sizeof(float)*mchan));
  }

  // Allocate memory for input bytes and header
  header=(char *) malloc(sizeof(char)*HEADERSIZE);
  if (dreamBeam == 0) {
    for (i=0;i<4;i++) {
      udpbuf[i]= reader->meta->outputData[i];
      checkCudaErrors(cudaMalloc((void **) &dudpbuf_c[i], (size_t) sizeof(char)*nsamp*nsub));
    }
  } else {
    for (i=0;i<4;i++) {
      udpbuf[i]= reader->meta->outputData[i];
      checkCudaErrors(cudaMalloc((void **) &dudpbuf_f[i], (size_t) sizeof(float)*nsamp*nsub));
    }
  }

  // Allocate output buffers
  fbuf=(float *) malloc(sizeof(float)*nsamp*nsub);
  checkCudaErrors(cudaMalloc((void **) &dfbuf, (size_t) sizeof(float)*nsamp*nsub));
  

  // Allocate final data product memories; differs based on wheter we are re-digitising before writing to disk or not
  if (redig) {
    for(i = 0; i < numStreams; i++)
      cbuf[i] = (unsigned char**) malloc(sizeof(unsigned char*)*ndm);
    for (i = 0; i < ndm; i++)
      for (j = 0; j < numStreams; j++)
        cbuf[j][i]=(unsigned char *) malloc(sizeof(unsigned char)*msamp*mchan/ndec);
    checkCudaErrors(cudaMalloc((void **) &dcbuf, (size_t) sizeof(unsigned char)*msamp*mchan/ndec));
  } else {
    for(i = 0; i < numStreams; i++)
      cbuff[i] = (float**) malloc(sizeof(float*)*ndm);
    for (i = 0; i < ndm; i++)
      for (j = 0; j < numStreams; j++)
        cbuff[j][i] = (float *) malloc(sizeof(float)*msamp*mchan/ndec);
    if (ndec > 1) checkCudaErrors(cudaMalloc((void **) &dcbuff, (size_t) sizeof(float)*msamp*mchan/ndec));
  }


  // Allocate DMs and copy to device
  dm=(float *) malloc(sizeof(float)*ndm);
  for (idm=0;idm<ndm;idm++)
    dm[idm]=dm_start+(float) idm*dm_step;
  checkCudaErrors(cudaMalloc((void **) &ddm, (size_t) sizeof(float)*ndm));
  checkCudaErrors(cudaMemcpy(ddm,dm,sizeof(float)*ndm,cudaMemcpyHostToDevice));

  // Allow memory alloation/copy actions to finish before processing
  cudaDeviceSynchronize();

  // Compute chirp
  blocksize.x=32; blocksize.y=32; blocksize.z=1;
  gridsize.x=nsub/blocksize.x+1; gridsize.y=nchan/blocksize.y+1; gridsize.z=ndm/blocksize.z+1;
  compute_chirp<<<gridsize,blocksize>>>(hdr.fcen,nsub*hdr.bwchan,ddm,nchan,nbin,nsub,ndm,dc);

  // Write temporary filterbank header
  file=fopen("/tmp/header.fil","w");
  if (file == NULL) {
    fprintf(stderr, "ERROR: Unable to open /tmp/header.fil to write temporary header; exiting.\n");
    exit(1);
  }
  write_filterbank_header(hdr,file);
  fclose(file);
  file=fopen("/tmp/header.fil","r");
  if (file == NULL) {
    fprintf(stderr, "ERROR: Unable to re-open /tmp/header.fil to read temporary header length; exiting.\n");
    exit(1);
  }
  bytes_read=fread(fheader,sizeof(char),1024,file);
  fclose(file);
  
  // Format file names and open
  outfile=(FILE **) malloc(sizeof(FILE *)*ndm);
  for (idm=0;idm<ndm;idm++) {
    sprintf(fname,"%s_cDM%06.2f_P%03d.fil",obsid,dm[idm],part);

    outfile[idm]=fopen(fname,"w");
    if (outfile[idm] == NULL) {
      fprintf(stderr, "Unable to open output file %s, exiting.\n", fname);
      exit(1);
    }
  }
  // Write headers
  for (idm=0;idm<ndm;idm++) {
    // Send header
    fwrite(fheader,sizeof(char),bytes_read,outfile[idm]);
  }


  // Loop over input file contents
  double timeInSeconds = 0.0;
  nread = INT_MAX;

  // Skip the first noverlap samples as they are 0'd
  int writeOffset = noverlap;
  int writeSize;


  // Remnants of an attempt to paralelise the I/O; leave in for now
  //int streamIdx = iblock % numStreams;
  int streamIdx = 0;
  cudaStream_t stream = streams[streamIdx];

  cufftSetStream(ftc2cf, stream);
  cufftSetStream(ftc2cb, stream);
  
  if (testmode)
    exit(0);

  CLICK(tick);

  float dt = 0.0;
  nread_tmp = reader->meta->packetsPerIteration * UDPNTIMESLICE;

  #pragma omp parallel default(shared)
  {
  #pragma omp single nowait
  {
  for (int iblock=0;;iblock++) {

    // Wait to finish reading in the next block
    #pragma omp taskwait


    if (nread > nread_tmp) {
      nread = nread_tmp;
    }
    // Determine the output length
    writeSize = (nread-writeOffset)*nsub/ndec;

    // Count up the total bytes read and calculate the read time
    total_ts_read += nread;
    printf("Block: %d: Read %ld MB in %.2f s\n",iblock,sizeof(char)*nread*nsub*4/(1<<20), dt);


    // Sanity check the read data size
    if (nread==0) {
      printf("No data read from last file; assuming EOF, finishng up.\n");
      break;
    } else if (iblock != 0 && nread < nread_tmp) {
      printf("Received less data than expected; we may have parsed out of order data or we are nearing the EOF.\n");
    }

    // Copy buffers to device, waiting for the previous overlap operation to finish first
    cudaStreamWaitEvent(stream, events[1], 0);

    CLICK(tick1);
    if (dreamBeam == 0) {
      for (i=0;i<4;i++) {
        checkCudaErrors(cudaMemcpyAsync(dudpbuf_c[i],udpbuf[i],sizeof(char)*nread*nsub,cudaMemcpyHostToDevice,stream));
      }
    } else {
      for (i=0;i<4;i++) {
        checkCudaErrors(cudaMemcpyAsync(dudpbuf_f[i],udpbuf[i],sizeof(float)*nread*nsub,cudaMemcpyHostToDevice,stream));
      }
    }

    printf("%ld, %ld\n", sizeof(float)*nread*nsub, reader->meta->packetOutputLength[0] * reader->meta->packetsPerIteration);
    cudaEventRecord(events[0], stream);

    printf("Tasking\n");
    // Start reading the next block of data, after the memcpy has finished
    #pragma omp task shared(reader, tick0, tock0, nread_tmp, events, dt)
    {
      printf("Tasked\n");
      // Hold the host execution until we can confirm the async memory transfer for the raw data has finished
      cudaEventSynchronize(events[0]);
      CLICK(tick0);
      #pragma omp atomic write
      nread_tmp = reshapeRawUdp(reader, checkinputs);
      CLICK(tock0);
      #pragma omp atomic write
      dt = TICKTOCK(tick0, tock0);
      printf("Task end\n");
    }
    printf("CUDA\n");

    // Unpack data and padd data
    blocksize.x=32; blocksize.y=32; blocksize.z=1;
    gridsize.x=nbin/blocksize.x+1; gridsize.y=nfft/blocksize.y+1; gridsize.z=nsub/blocksize.z+1;
    if (dreamBeam == 0) {
      if (iblock > 0) {
        unpack_and_padd<char><<<gridsize,blocksize,0,stream>>>(dudpbuf_c[0],dudpbuf_c[1],dudpbuf_c[2],dudpbuf_c[3],nread,nbin,nfft,nsub,noverlap,cp1p,cp2p);
      } else {
        unpack_and_padd_first_iteration<char><<<gridsize,blocksize,0,stream>>>(dudpbuf_c[0],dudpbuf_c[1],dudpbuf_c[2],dudpbuf_c[3],nread,nbin,nfft,nsub,noverlap,cp1p,cp2p);
      }
    } else {
      if (iblock > 0) {
        unpack_and_padd<float><<<gridsize,blocksize,0,stream>>>(dudpbuf_f[0],dudpbuf_f[1],dudpbuf_f[2],dudpbuf_f[3],nread,nbin,nfft,nsub,noverlap,cp1p,cp2p);
      } else {
        unpack_and_padd_first_iteration<float><<<gridsize,blocksize,0,stream>>>(dudpbuf_f[0],dudpbuf_f[1],dudpbuf_f[2],dudpbuf_f[3],nread,nbin,nfft,nsub,noverlap,cp1p,cp2p);
      }

    }

    // Perform FFTs
    cudaStreamWaitEvent(stream, events[2], 0);
    //cufftSetStream(ftc2cf, stream);

    checkCudaErrors(cufftExecC2C(ftc2cf,(cufftComplex *) cp1p,(cufftComplex *) cp1p,CUFFT_FORWARD));
    checkCudaErrors(cufftExecC2C(ftc2cf,(cufftComplex *) cp2p,(cufftComplex *) cp2p,CUFFT_FORWARD));

    // Swap spectrum halves for large FFTs
    blocksize.x=32; blocksize.y=32; blocksize.z=1;
    gridsize.x=nbin/blocksize.x+1; gridsize.y=nfft*nsub/blocksize.y+1; gridsize.z=1;
    swap_spectrum_halves<<<gridsize,blocksize,0,stream>>>(cp1p,cp2p,nbin,nfft*nsub);

    // Swap the cuFFT operation to the current stream
    //cufftSetStream(ftc2cb, stream);
 
    // Loop over dms
    for (idm=0;idm<ndm;idm++) {

      // Perform complex multiplication of FFT'ed data with chirp
      blocksize.x=32; blocksize.y=32; blocksize.z=1;
      gridsize.x=nbin*nsub/blocksize.x+1; gridsize.y=nfft/blocksize.y+1; gridsize.z=1;
      PointwiseComplexMultiply<<<gridsize,blocksize,0,stream>>>(cp1p,dc,cp1,nbin*nsub,nfft,idm,1.0/(float) nbin);
      PointwiseComplexMultiply<<<gridsize,blocksize,0,stream>>>(cp2p,dc,cp2,nbin*nsub,nfft,idm,1.0/(float) nbin);
      
      // When cp1/2p are no longer needed, start overlapping the data for the next iteration on a separate stream
      if (idm == ndm - 1) {
        cudaEventRecord(events[2], stream);
        cudaStreamWaitEvent(streams[numStreams], events[2], 0);
        blocksize.x=32; blocksize.y=32; blocksize.z=1;
        gridsize.x=nbin/blocksize.x+1; gridsize.y=nfft/blocksize.y+1; gridsize.z=nsub/blocksize.z+1;
        if (dreamBeam == 0) {
          padd_next_iteration<char><<<gridsize,blocksize,0,streams[numStreams]>>>(dudpbuf_c[0],dudpbuf_c[1],dudpbuf_c[2],dudpbuf_c[3],nread,nbin,nfft,nsub,noverlap,cp1p,cp2p);
        } else {
          padd_next_iteration<float><<<gridsize,blocksize,0,streams[numStreams]>>>(dudpbuf_f[0],dudpbuf_f[1],dudpbuf_f[2],dudpbuf_f[3],nread,nbin,nfft,nsub,noverlap,cp1p,cp2p);
        }
        cudaEventRecord(events[1], streams[numStreams]);
      }
      // Swap spectrum halves for small FFTs
      blocksize.x=32; blocksize.y=32; blocksize.z=1;
      gridsize.x=mbin/blocksize.x+1; gridsize.y=nchan*nfft*nsub/blocksize.y+1; gridsize.z=1;
      swap_spectrum_halves<<<gridsize,blocksize,0,stream>>>(cp1,cp2,mbin,nchan*nfft*nsub);
      
      // Perform FFTs
      checkCudaErrors(cufftExecC2C(ftc2cb,(cufftComplex *) cp1,(cufftComplex *) cp1,CUFFT_INVERSE));
      checkCudaErrors(cufftExecC2C(ftc2cb,(cufftComplex *) cp2,(cufftComplex *) cp2,CUFFT_INVERSE));

      // Wait for the previous memory transfer to finish
      cudaStreamWaitEvent(stream, dmWriteEvents[streamIdx][idm-1 > -1 ? idm-1 : ndm - 1], 0);
      // Detect data
      blocksize.x=32; blocksize.y=32; blocksize.z=1;
      gridsize.x=mbin/blocksize.x+1; gridsize.y=nchan/blocksize.y+1; gridsize.z=nfft/blocksize.z+1;
      transpose_unpadd_and_detect<<<gridsize,blocksize,0,stream>>>(cp1,cp2,mbin,nchan,nfft,nsub,noverlap/nchan,nread/nchan,dfbuf);

      if (redig) {
        // Compute block sums for redigitization
        blocksize.x=32; blocksize.y=32; blocksize.z=1;
        gridsize.x=mchan/blocksize.x+1; gridsize.y=mblock/blocksize.y+1; gridsize.z=1;
        compute_block_sums<<<gridsize,blocksize,0,stream>>>(dfbuf,mchan,mblock,msum,bs1,bs2);
        
        // Compute channel stats
        blocksize.x=32; blocksize.y=1; blocksize.z=1;
        gridsize.x=mchan/blocksize.x+1; gridsize.y=1; gridsize.z=1;
        compute_channel_statistics<<<gridsize,blocksize,0,stream>>>(mchan,mblock,msum,bs1,bs2,zavg,zstd);

        // Redigitize data to 8bits
        blocksize.x=32; blocksize.y=32; blocksize.z=1;
        gridsize.x=mchan/blocksize.x+1; gridsize.y=mblock/blocksize.y+1; gridsize.z=1;
        if (ndec==1)
    redigitize<<<gridsize,blocksize,0,stream>>>(dfbuf,mchan,mblock,msum,zavg,zstd,3.0,5.0,dcbuf);
        else
    decimate_and_redigitize<<<gridsize,blocksize,0,stream>>>(dfbuf,ndec,mchan,mblock,msum,zavg,zstd,3.0,5.0,dcbuf);      

        // Copy buffer to host
        checkCudaErrors(cudaMemcpyAsync(cbuf[streamIdx][idm],dcbuf,sizeof(unsigned char)*msamp*mchan/ndec,cudaMemcpyDeviceToHost,stream));

      } else {
        if (ndec==1) {
          checkCudaErrors(cudaMemcpyAsync(cbuff[streamIdx][idm], dfbuf,sizeof(float)*msamp*mchan,cudaMemcpyDeviceToHost,stream));
        } else {
          blocksize.x=32; blocksize.y=32; blocksize.z=1;
          gridsize.x=mchan/blocksize.x+1; gridsize.y=mblock/blocksize.y+1; gridsize.z=1;
          decimate<<<gridsize,blocksize,0,stream>>>(dfbuf,ndec,mchan,mblock,msum,dcbuff);
          checkCudaErrors(cudaMemcpyAsync(cbuff[streamIdx][idm],dcbuff,sizeof(float)*msamp*mchan/ndec,cudaMemcpyDeviceToHost,stream));
        }
        

      }

      // Record when the final memcpy finishes to block disk writes
      cudaEventRecord(dmWriteEvents[streamIdx][idm], stream);
    }

    // Wrtie results to disk, waiting for each DM's memcpy to finish first
    for (idm=0;idm<ndm;idm++) {
      if (redig) {
        write_to_disk_char(&(cbuf[streamIdx][idm][writeOffset*nsub/ndec]), &(outfile[idm]), writeSize, &(dmWriteEvents[streamIdx][idm]));
      } else {
        write_to_disk_float(&(cbuff[streamIdx][idm][writeOffset*nsub/ndec]), &(outfile[idm]), writeSize, &(dmWriteEvents[streamIdx][idm]));
      }
    }


    CLICK(tock);
    printf("Processed %d DMs in %.2f s\n",ndm, TICKTOCK(tick1, tock));
    timeInSeconds += (double) (nread - writeOffset) * timeOffset;
    elapsedTime = (double) TICKTOCK(tick, tock);
    printf("Current data processed: %02ld:%02ld:%05.2lf (%1.2lfs) in %1.2lf seconds (%1.2lf/s)\n\n", (long int) (timeInSeconds / 3600.0), (long int) ((fmod(timeInSeconds, 3600.0)) / 60.0), fmod(timeInSeconds, 60.0), timeInSeconds, elapsedTime, (double) timeInSeconds / elapsedTime);

    // Exit when we pass the read length limit
    if (total_ts_read > ts_read) {
      break;
    }

    if (iblock == 0) {
      writeOffset = 0;
    }

  }
  // Parallel
  }
  // Single nowait
  }


  CLICK(tock);
  printf("Finished processing %lfs of data in %fs (%lf/s). Cleaning up...\n", timeInSeconds, TICKTOCK(tick, tock), (float) timeInSeconds / (float) TICKTOCK(tick, tock));


  //omp_destroy_lock(&readLock);
  // Close files
  printf("Closing files...\n");
  for (i=0;i<ndm;i++)
    fclose(outfile[i]);

  // Reader cleanup
  printf("Cleaning up the reader...\n");
  lofar_udp_reader_cleanup(reader);

  // Free
  printf("Free the header...\n");
  free(header);
  printf("Free the GPU dedbuff\n");
  if (dreamBeam == 0) {
    for (i=0;i<4;i++) {
      cudaFree(dudpbuf_c[i]);  
    }
  } else {
    for (i=0;i<4;i++) {
      cudaFree(dudpbuf_f[i]);  
    }
  }
  printf("Free host data...\n");
  free(fbuf);
  free(dm);
  free(outfile);

  printf("Free deci/output components\n");
  if (redig) {
    for (i = 0; i < ndm; i++)
      for (j =0; j < numStreams; j++)
        free(cbuf[j][i]);
    cudaFree(bs1);
    cudaFree(bs2);
    cudaFree(zavg);
    cudaFree(zstd);
    cudaFree(dcbuf);
  } else {
    printf("Loopy\n");
    for (j =0; j < numStreams; j++)
      for (i = 0; i < ndm; i++)
        free(cbuff[j][i]);
      free(cbuff[j]);    
    printf("Deci\n");
    if (ndec > 1) cudaFree(dcbuff);
  }


  printf("Cuda workspace...\n");
  cudaFree(cufftWorkArea);
  cudaFree(dfbuf);
  cudaFree(cp1);
  cudaFree(cp2);
  cudaFree(cp1p);
  cudaFree(cp2p);
  cudaFree(dc);
  cudaFree(ddm);

  // Free plan
  printf("cuFFT....\n");
  cufftDestroy(ftc2cf);
  cufftDestroy(ftc2cb);

  printf("Streams...\n");
  for(i = 0; i < numStreams + 1; i++)
    cudaStreamDestroy(streams[i]);
  printf("Events....\n");
  for(i = 0; i < numEvents; i++)
    cudaEventDestroy(events[i]);

  printf("DM events...\n");
  for (i = 0; i < ndm; i++)
    for (j =0; j < numStreams; j++)
      cudaEventDestroy(dmWriteEvents[j][i]);

  return 0;
}


void write_to_disk_float(float* outputArray, FILE** outputFile, int nsamples, cudaEvent_t* waitEvent)
{
  cudaEventSynchronize(*waitEvent);
  fwrite(outputArray,sizeof(float),nsamples, *outputFile); 
}

void write_to_disk_char(unsigned char* outputArray, FILE** outputFile, int nsamples, cudaEvent_t* waitEvent)
{
  cudaEventSynchronize(*waitEvent);
  fwrite(outputArray,sizeof(char),nsamples, *outputFile); 
}



// Rip out sigproc's header reader. Don't have the time to spend several hours reimplementing it; all credit to Lorimer et al.
//BEGIN SIGPROC READ_HEADER.C
//
int strings_equal (char *string1, char *string2) /* includefile */
{
  if (!strcmp(string1,string2)) {
    return 1;
  } else {
    return 0;
  }
}
/* read a string from the input which looks like nchars-char[1-nchars] */
void get_string(FILE *inputfile, int *nbytes, char string[])
{
  int nchar;
  strcpy(string,"ERROR");
  if (! fread(&nchar, sizeof(int), 1, inputfile)) fprintf(stderr, "Failed to get int at %d\n", *nbytes);
  *nbytes=sizeof(int);
  if (feof(inputfile)) exit(0);
  if (nchar>80 || nchar<1) return;
  if (! fread(string, nchar, 1, inputfile)) fprintf(stderr, "Failed to get stirng at %d\n", *nbytes);
  string[nchar]='\0';
  *nbytes+=nchar;
}

/* attempt to read in the general header info from a pulsar data file */
struct header read_header(FILE *inputfile) /* includefile */
{
  char string[80], message[80];
  int nbytes,totalbytes,expecting_rawdatafile=0,expecting_source_name=0; 
  int isign=0, dummyread=0;
  struct header hdr;


  /* try to read in the first line of the header */
  get_string(inputfile,&nbytes,string);
  if (!strings_equal(string, (char *) "HEADER_START")) {
  /* the data file is not in standard format, rewind and return */
  rewind(inputfile);
  fprintf(stderr, "Unexpected input header; exiting.");
  exit(1);
  }
  /* store total number of bytes read so far */
  totalbytes=nbytes;

  /* loop over and read remaining header lines until HEADER_END reached */
  // David McKenna: We don't need all of these; ignore those values and just reference their lengths
  while (1) {
    get_string(inputfile,&nbytes,string);
    if (strings_equal(string, (char *) "HEADER_END")) break;
    totalbytes+=nbytes;
    if (strings_equal(string, (char *) "rawdatafile")) {
      expecting_rawdatafile=1;
    } else if (strings_equal(string, (char *) "source_name")) {
      expecting_source_name=1;
    } else if (strings_equal(string, (char *) "FREQUENCY_START")) {
      // pass
    } else if (strings_equal(string, (char *) "FREQUENCY_END")) {
      // pass
    } else if (strings_equal(string, (char *) "az_start")) {
      fseek(inputfile, sizeof(double), SEEK_CUR);
      totalbytes+=sizeof(double);
    } else if (strings_equal(string, (char *) "za_start")) {
      fseek(inputfile, sizeof(double), SEEK_CUR);
      totalbytes+=sizeof(double);
    } else if (strings_equal(string, (char *) "src_raj")) {
      dummyread = fread(&(hdr.src_raj),sizeof(hdr.src_raj),1,inputfile);
      totalbytes+=sizeof(hdr.src_raj);
    } else if (strings_equal(string, (char *) "src_dej")) {
      dummyread = fread(&(hdr.src_dej),sizeof(hdr.src_dej),1,inputfile);
      totalbytes+=sizeof(hdr.src_dej);
    } else if (strings_equal(string, (char *) "tstart")) {
      dummyread = fread(&(hdr.tstart),sizeof(hdr.tstart),1,inputfile);
      totalbytes+=sizeof(hdr.tstart);
    } else if (strings_equal(string, (char *) "tsamp")) {
      dummyread = fread(&(hdr.tsamp),sizeof(hdr.tsamp),1,inputfile);
      totalbytes+=sizeof(hdr.tsamp);
    } else if (strings_equal(string, (char *) "period")) {
      fseek(inputfile, sizeof(double), SEEK_CUR);
      totalbytes+=sizeof(double);
    } else if (strings_equal(string, (char *) "fch1")) {
      dummyread = fread(&(hdr.fch1),sizeof(hdr.fch1),1,inputfile);
      totalbytes+=sizeof(hdr.fch1);
    } else if (strings_equal(string, (char *) "fchannel")) {
      fseek(inputfile, sizeof(double), SEEK_CUR);
      totalbytes+=sizeof(double);
    } else if (strings_equal(string, (char *) "foff")) {
      dummyread = fread(&(hdr.foff),sizeof(hdr.foff),1,inputfile);
      totalbytes+=sizeof(hdr.foff);
    } else if (strings_equal(string, (char *) "nchans")) {
      // nsub seems to be nchans in the sigproc hdr
      dummyread = fread(&(hdr.nsub),sizeof(hdr.nsub),1,inputfile);
      totalbytes+=sizeof(hdr.nsub);
    } else if (strings_equal(string, (char *) "telescope_id")) {
      dummyread = fread(&(hdr.tel),sizeof(hdr.tel),1,inputfile);
      totalbytes+=sizeof(hdr.tel);
    } else if (strings_equal(string, (char *) "machine_id")) {
      dummyread = fread(&(hdr.mach),sizeof(hdr.mach),1,inputfile);
      totalbytes+=sizeof(int);
    } else if (strings_equal(string, (char *) "data_type")) {
      fseek(inputfile, sizeof(int), SEEK_CUR);
      totalbytes+=sizeof(int);
    } else if (strings_equal(string, (char *) "ibeam")) {
      fseek(inputfile, sizeof(int), SEEK_CUR);
      totalbytes+=sizeof(int);
    } else if (strings_equal(string, (char *) "nbeams")) {
      fseek(inputfile, sizeof(int), SEEK_CUR);
      totalbytes+=sizeof(int);
    } else if (strings_equal(string, (char *) "nbits")) {
      dummyread = fread(&(hdr.nbit),sizeof(hdr.nbit),1,inputfile);
      totalbytes+=sizeof(hdr.nbit);
    } else if (strings_equal(string, (char *) "barycentric")) {
      fseek(inputfile, sizeof(int), SEEK_CUR);
      totalbytes+=sizeof(int);
    } else if (strings_equal(string, (char *) "pulsarcentric")) {
      fseek(inputfile, sizeof(int), SEEK_CUR);
      totalbytes+=sizeof(int);
    } else if (strings_equal(string, (char *) "nbins")) {
      fseek(inputfile, sizeof(int), SEEK_CUR);
      totalbytes+=sizeof(int);
    } else if (strings_equal(string, (char *) "nsamples")) {
      /* read this one only for backwards compatibility */
      fseek(inputfile, sizeof(int), SEEK_CUR);
      totalbytes+=sizeof(int);
    } else if (strings_equal(string, (char *) "nifs")) {
      fseek(inputfile, sizeof(int), SEEK_CUR);
      totalbytes+=sizeof(int);
    } else if (strings_equal(string, (char *) "npuls")) {
      totalbytes+=sizeof(long int);
    } else if (strings_equal(string, (char *) "refdm")) {
      fseek(inputfile, sizeof(double), SEEK_CUR);
      totalbytes+=sizeof(double);
    } else if (strings_equal(string, (char *) "signed")) {
      dummyread = fread(&isign,sizeof(isign),1,inputfile);
      totalbytes+=sizeof(isign);
    } else if (expecting_rawdatafile) {
      //strcpy(hdr.rawfname,string);
      expecting_rawdatafile=0;
    } else if (expecting_source_name) {
      strcpy(hdr.source_name,string);
      expecting_source_name=0;
    } else {
      sprintf(message,"read_header (%d) - unknown parameter: %s\n", dummyread, string);
      fprintf(stderr,"ERROR: %s\n",message);
      exit(1);
    } 
    if (totalbytes != ftell(inputfile)){
      fprintf(stderr,"ERROR: Header bytes does not equal file position\n");
      fprintf(stderr,"String was: '%s'\n",string);
      fprintf(stderr,"       header: %d file: %ld\n",totalbytes,ftell(inputfile));
      exit(1);
    }


  } 

  /* add on last header string */
  totalbytes+=nbytes;

  if (totalbytes != ftell(inputfile)){
    fprintf(stderr,"ERROR: Header bytes does not equal file position\n");
    fprintf(stderr,"       header: %d file: %ld\n",totalbytes,ftell(inputfile));
    exit(1);
  }

  /* return total number of bytes read */
  return hdr;
}
// END SIGPROC READ_HEADER.c




struct header read_sigproc_header(char *fname, char *dataname, int ports)
{

  FILE *tmpf;

  tmpf = fopen(fname, "r");
  if (tmpf == NULL) {
    fprintf(stderr, "Unable to open sigproc header at %s; exiting.\n", fname);
    exit(1);
  }
  struct header hdr = read_header(tmpf);
  fclose(tmpf);



  hdr.fcen = hdr.fch1 + (hdr.foff * hdr.nsub * 0.5);
  hdr.bwchan = fabs(hdr.foff);

 return hdr;
}

// Scale cufftComplex 
static __device__ __host__ inline cufftComplex ComplexScale(cufftComplex a,float s)
{
  cufftComplex c;
  c.x=s*a.x;
  c.y=s*a.y;
  return c;
}

// Complex multiplication
static __device__ __host__ inline cufftComplex ComplexMul(cufftComplex a,cufftComplex b)
{
  cufftComplex c;
  c.x=a.x*b.x-a.y*b.y;
  c.y=a.x*b.y+a.y*b.x;
  return c;
}

// Pointwise complex multiplication (and scaling)
static __global__ void PointwiseComplexMultiply(cufftComplex *a,cufftComplex *b,cufftComplex *c,int nx,int ny,int l,float scale)
{
  int i,j,k;
  i=blockIdx.x*blockDim.x+threadIdx.x;
  j=blockIdx.y*blockDim.y+threadIdx.y;

  if (i<nx && j<ny) {
    k=i+nx*j;
    c[k]=ComplexScale(ComplexMul(a[k],b[i+nx*l]),scale);
  }
}

// Compute chirp
__global__ void compute_chirp(double fcen,double bw,float *dm,int nchan,int nbin,int nsub,int ndm,cufftComplex *c)
{
  int ibin,ichan,isub,idm,mbin,idx;
  double s,rt,t,f,fsub,fchan,bwchan,bwsub;

  // Number of channels per subband
  mbin=nbin/nchan;

  // Subband bandwidth
  bwsub=bw/nsub;

  // Channel bandwidth
  bwchan=bw/(nchan*nsub);

  // Indices of input data
  isub=blockIdx.x*blockDim.x+threadIdx.x;
  ichan=blockIdx.y*blockDim.y+threadIdx.y;
  idm=blockIdx.z*blockDim.z+threadIdx.z;

  // Keep in range
  if (isub<nsub && ichan<nchan && idm<ndm) {
    // Main constant
    s=2.0*M_PI*dm[idm]/DMCONSTANT;

    // Frequencies
    fsub=fcen-0.5*bw+bw*(float) isub/(float) nsub+0.5*bw/(float) nsub;
    fchan=fsub-0.5*bwsub+bwsub*(float) ichan/(float) nchan+0.5*bwsub/(float) nchan;
      
    // Loop over bins in channel
    for (ibin=0;ibin<mbin;ibin++) {
      // Bin frequency
      f=-0.5*bwchan+bwchan*(float) ibin/(float) mbin+0.5*bwchan/(float) mbin;
      
      // Phase delay
      rt=-f*f*s/((fchan+f)*fchan*fchan);
      
      // Taper
      t=1.0/sqrt(1.0+pow((f/(0.47*bwchan)),80));
      
      // Index
      idx=ibin+ichan*mbin+isub*mbin*nchan+idm*nsub*mbin*nchan;
      
      // Chirp
      c[idx].x=cos(rt)*t;
      c[idx].y=sin(rt)*t;
    }
  }

  return;
}

// Unpack the input buffer and generate complex timeseries. The output
// timeseries are padded with noverlap samples on either side for the
// convolution.
template<typename I> __global__ void unpack_and_padd(I *dbuf0,I *dbuf1,I *dbuf2,I *dbuf3,int nsamp,int nbin,int nfft,int nsub,int noverlap,cufftComplex *cp1,cufftComplex *cp2)
{
  int64_t ibin,ifft,isamp,isub,idx1,idx2;

  // Indices of input data
  ibin=blockIdx.x*blockDim.x+threadIdx.x;
  ifft=blockIdx.y*blockDim.y+threadIdx.y;
  isub=blockIdx.z*blockDim.z+threadIdx.z;

  // Only compute valid threads
  if (ibin<nbin && ifft<nfft && isub<nsub) {
    isamp=ibin+(nbin-2*noverlap)*ifft-noverlap;
    if (isamp >= noverlap) {
      idx1=ibin+nbin*isub+nsub*nbin*ifft;
      idx2=isub+nsub*(isamp-noverlap);
      cp1[idx1].x=(float) dbuf0[idx2];
      cp1[idx1].y=(float) dbuf1[idx2];
      cp2[idx1].x=(float) dbuf2[idx2];
      cp2[idx1].y=(float) dbuf3[idx2];
    }
  }

  return;
}

// Unpack the input buffer and generate complex timeseries. The output
// timeseries are padded with noverlap samples on either side for the
// convolution. This is separate from the main kernel to minimise performance
// loss to branching on the GPU. On the first iteration, we want to fill
// the final non-noverlap region and final noverlap region so that they can 
// match the first noverlap region and first non-noverlap on the second
// iteration
template<typename I> __global__ void unpack_and_padd_first_iteration(I *dbuf0,I *dbuf1,I *dbuf2,I *dbuf3,int nsamp,int nbin,int nfft,int nsub,int noverlap,cufftComplex *cp1,cufftComplex *cp2)
{
  int64_t ibin,ifft,isamp,isub,idx1,idx2;

  // Indices of input data
  ibin=blockIdx.x*blockDim.x+threadIdx.x;
  ifft=blockIdx.y*blockDim.y+threadIdx.y;
  isub=blockIdx.z*blockDim.z+threadIdx.z;

  // Only compute valid threads
  if (ibin<nbin && ifft<nfft && isub<nsub) {
    isamp=ibin+(nbin-2*noverlap)*ifft-noverlap;
    if (isamp >= noverlap) {
      idx1=ibin+nbin*isub+nsub*nbin*ifft;
      idx2=isub+nsub*(isamp-noverlap);

      cp1[idx1].x=(float) dbuf0[idx2];
      cp1[idx1].y=(float) dbuf1[idx2];
      cp2[idx1].x=(float) dbuf2[idx2];
      cp2[idx1].y=(float) dbuf3[idx2];
    } else if (isamp > -noverlap) {
      idx1=ibin+nbin*isub+nsub*nbin*ifft;
      idx2=isub+nsub*(noverlap-isamp);

      cp1[idx1].x=(float) dbuf0[idx2];
      cp1[idx1].y=(float) dbuf1[idx2];
      cp2[idx1].x=(float) dbuf2[idx2];
      cp2[idx1].y=(float) dbuf3[idx2];
    }
  }

  return;
}

// Unpack the input buffer and generate complex timeseries. The output
// timeseries are located in the first noverlap region and first non-
// noverlap region, for continuous time series between data blocks
// 
// overlap_(timeblock)_(index)
// t = 0: overlap_0_0: nfft_0_0, nfft_0_1... nfft_0_N-1, nfft_0 N: overlap_0_1
// t = 1: nfft_0_N: overlap_0_1, nfft_1_0.... nfft_1_N-1:overlap_1_1
// t = 2 nfft_1_N-1: overlap_1_1...
// etc
template<typename I> __global__ void padd_next_iteration(I *dbuf0,I *dbuf1,I *dbuf2,I *dbuf3,int nsamp,int nbin,int nfft,int nsub,int noverlap,cufftComplex *cp1,cufftComplex *cp2)
{
  int64_t ibin,ifft,isamp,isub,idx1,idx2;

  // Indices of input data
  ibin=blockIdx.x*blockDim.x+threadIdx.x;
  ifft=blockIdx.y*blockDim.y+threadIdx.y;
  isub=blockIdx.z*blockDim.z+threadIdx.z;

  // Only compute valid threads
  if (ibin<nbin && ifft<nfft && isub<nsub) {
    isamp=ibin+(nbin-2*noverlap)*ifft;
    if (isamp<2*noverlap) {
      idx1=ibin+nbin*isub+nsub*nbin*ifft;
      idx2=isub+nsub*(isamp+nsamp-2*noverlap);
      cp1[idx1].x=(float) dbuf0[idx2];
      cp1[idx1].y=(float) dbuf1[idx2];
      cp2[idx1].x=(float) dbuf2[idx2];
      cp2[idx1].y=(float) dbuf3[idx2];
    }
  }
}


// Since complex-to-complex FFTs put the center frequency at bin zero
// in the frequency domain, the two halves of the spectrum need to be
// swapped.
__global__ void swap_spectrum_halves(cufftComplex *cp1,cufftComplex *cp2,int nx,int ny)
{
  int64_t i,j,k,l,m;
  cufftComplex tp1,tp2;

  i=blockIdx.x*blockDim.x+threadIdx.x;
  j=blockIdx.y*blockDim.y+threadIdx.y;
  if (i<nx/2 && j<ny) {
    if (i<nx/2)
      k=i+nx/2;
    else
      k=i-nx/2;
    l=i+nx*j;
    m=k+nx*j;
    tp1.x=cp1[l].x;
    tp1.y=cp1[l].y;
    tp2.x=cp2[l].x;
    tp2.y=cp2[l].y;
    cp1[l].x=cp1[m].x;
    cp1[l].y=cp1[m].y;
    cp2[l].x=cp2[m].x;
    cp2[l].y=cp2[m].y;
    cp1[m].x=tp1.x;
    cp1[m].y=tp1.y;
    cp2[m].x=tp2.x;
    cp2[m].y=tp2.y;
  }

  return;
}

// After the segmented FFT the data is in a cube of nbin by nchan by
// nfft, where nbin and nfft are the time indices. Here we rearrange
// the 3D data cube into a 2D array of frequency and time, while also
// removing the overlap regions and detecting (generating Stokes I).
__global__ void transpose_unpadd_and_detect(cufftComplex *cp1,cufftComplex *cp2,int nbin,int nchan,int nfft,int nsub,int noverlap,int nsamp,float *fbuf)
{
  int64_t ibin,ichan,ifft,isub,isamp,idx1,idx2;
  
  ibin=blockIdx.x*blockDim.x+threadIdx.x;
  ichan=blockIdx.y*blockDim.y+threadIdx.y;
  ifft=blockIdx.z*blockDim.z+threadIdx.z;
  if (ibin<nbin && ichan<nchan && ifft<nfft) {
    // Loop over subbands
    for (isub=0;isub<nsub;isub++) {
      // Padded array index
      //      idx1=ibin+nbin*isub+nsub*nbin*(ichan+nchan*ifft);
      idx1=ibin+ichan*nbin+(nsub-isub-1)*nbin*nchan+ifft*nbin*nchan*nsub;

      // Time index
      isamp=ibin+(nbin-2*noverlap)*ifft-noverlap;
      
      // Output array index
      idx2=(nchan-ichan-1)+isub*nchan+nsub*nchan*isamp;
      
      // Select data points from valid region
      if (ibin>=noverlap && ibin<=nbin-noverlap && isamp>=0 && isamp<nsamp)
  fbuf[idx2]=cp1[idx1].x*cp1[idx1].x+cp1[idx1].y*cp1[idx1].y+cp2[idx1].x*cp2[idx1].x+cp2[idx1].y*cp2[idx1].y;
    }
  }

  return;
}

void send_string(const char *string,FILE *file)
{
  int len, lenoff = 0;

  len=strlen(string);
  if (len > 63) {
    lenoff = len - 64;
    len = 64;
  }
  fwrite(&len,sizeof(int),1,file);
  fwrite(&(string[lenoff]),sizeof(char),len,file);

  return;
}

void send_float(const char *string,float x,FILE *file)
{
  send_string(string,file);
  fwrite(&x,sizeof(float),1,file);

  return;
}

void send_int(const char *string,int x,FILE *file)
{
  send_string(string,file);
  fwrite(&x,sizeof(int),1,file);

  return;
}

void send_double(const char *string,double x,FILE *file)
{
  send_string(string,file);
  fwrite(&x,sizeof(double),1,file);

  return;
}

double dec2sex(double x)
{
  double d,sec,min,deg;
  char sign;
  char tmp[32];

  sign=(x<0 ? '-' : ' ');
  x=3600.0*fabs(x);

  sec=fmod(x,60.0);
  x=(x-sec)/60.0;
  min=fmod(x,60.0);
  x=(x-min)/60.0;
  deg=x;

  sprintf(tmp,"%c%02d%02d%09.6lf",sign,(int) deg,(int) min,sec);
  sscanf(tmp,"%lf",&d);

  return d;
}

void write_filterbank_header(struct header h,FILE *file)
{
  //double ra,de;


  //ra=dec2sex(h.src_raj/15.0);
  //de=dec2sex(h.src_dej);
  
  send_string("HEADER_START",file);
  send_string("rawdatafile",file);
  send_string(h.rawfname[0],file);
  send_string("source_name",file);
  send_string(h.source_name,file);
  send_int("machine_id",h.mach,file);
  send_int("telescope_id",h.tel,file);
  send_double("src_raj",h.src_raj,file);
  send_double("src_dej",h.src_dej,file);
  send_int("data_type",1,file);
  send_double("fch1",h.fch1,file);
  send_double("foff",h.foff,file);
  send_int("nchans",h.nchan,file);
  send_int("nbeams",0,file);
  send_int("ibeam",0,file);
  send_int("nbits",h.nbit,file);
  send_double("tstart",h.tstart,file);
  send_double("tsamp",h.tsamp,file);
  send_int("nifs",1,file);
  send_string("HEADER_END",file);

  return;
}

// Compute segmented sums for later computation of offset and scale
__global__ void compute_block_sums(float *z,int nchan,int nblock,int nsum,float *bs1,float *bs2)
{
  int64_t ichan,iblock,isum,idx1,idx2;

  ichan=blockIdx.x*blockDim.x+threadIdx.x;
  iblock=blockIdx.y*blockDim.y+threadIdx.y;
  if (ichan<nchan && iblock<nblock) {
    idx1=ichan+nchan*iblock;
    bs1[idx1]=0.0;
    bs2[idx1]=0.0;
    for (isum=0;isum<nsum;isum++) {
      idx2=ichan+nchan*(isum+iblock*nsum);
      bs1[idx1]+=z[idx2];
      bs2[idx1]+=z[idx2]*z[idx2];
    }
  }

  return;
}

// Compute segmented sums for later computation of offset and scale
__global__ void compute_channel_statistics(int nchan,int nblock,int nsum,float *bs1,float *bs2,float *zavg,float *zstd)
{
  int64_t ichan,iblock,idx1;
  double s1,s2;

  ichan=blockIdx.x*blockDim.x+threadIdx.x;
  if (ichan<nchan) {
    s1=0.0;
    s2=0.0;
    for (iblock=0;iblock<nblock;iblock++) {
      idx1=ichan+nchan*iblock;
      s1+=bs1[idx1];
      s2+=bs2[idx1];
    }
    zavg[ichan]=s1/(float) (nblock*nsum);
    zstd[ichan]=s2/(float) (nblock*nsum)-zavg[ichan]*zavg[ichan];
    zstd[ichan]=sqrt(zstd[ichan]);
  }

  return;
}

// Redigitize the filterbank to 8 bits in segments
__global__ void redigitize(float *z,int nchan,int nblock,int nsum,float *zavg,float *zstd,float zmin,float zmax,unsigned char *cz)
{
  int64_t ichan,iblock,isum,idx1;
  float zoffset,zscale;

  ichan=blockIdx.x*blockDim.x+threadIdx.x;
  iblock=blockIdx.y*blockDim.y+threadIdx.y;
  if (ichan<nchan && iblock<nblock) {
    zoffset=zavg[ichan]-zmin*zstd[ichan];
    zscale=(zmin+zmax)*zstd[ichan];

    for (isum=0;isum<nsum;isum++) {
      idx1=ichan+nchan*(isum+iblock*nsum);
      z[idx1]-=zoffset;
      z[idx1]*=256.0/zscale;
      cz[idx1]=(unsigned char) z[idx1];
      if (z[idx1]<0.0) cz[idx1]=0;
      if (z[idx1]>255.0) cz[idx1]=255;
    }
  }

  return;
}

// Decimate and Redigitize the filterbank to 8 bits in segments
__global__ void decimate_and_redigitize(float *z,int ndec,int nchan,int nblock,int nsum,float *zavg,float *zstd,float zmin,float zmax,unsigned char *cz)
{
  int64_t ichan,iblock,isum,idx1,idx2,idec;
  float zoffset,zscale,ztmp;

  ichan=blockIdx.x*blockDim.x+threadIdx.x;
  iblock=blockIdx.y*blockDim.y+threadIdx.y;
  if (ichan<nchan && iblock<nblock) {
    zoffset=zavg[ichan]-zmin*zstd[ichan];
    zscale=(zmin+zmax)*zstd[ichan];

    for (isum=0;isum<nsum;isum+=ndec) {
      idx2=ichan+nchan*(isum/ndec+iblock*nsum/ndec);
      for (idec=0,ztmp=0.0;idec<ndec;idec++) {
  idx1=ichan+nchan*(isum+idec+iblock*nsum);
  ztmp+=z[idx1];
      }
      ztmp/=(float) ndec;
      ztmp-=zoffset;
      ztmp*=256.0/zscale;
      cz[idx2]=(unsigned char) ztmp;
      if (ztmp<0.0) cz[idx2]=0;
      if (ztmp>255.0) cz[idx2]=255;
    }
  }

  return;
}


// Decimate the filterbank without redigitisation
__global__ void decimate(float *z,int ndec,int nchan,int nblock,int nsum,float *cz)
{
  int64_t ichan,iblock,isum,idx1,idx2,idec;
  float ztmp;

  ichan=blockIdx.x*blockDim.x+threadIdx.x;
  iblock=blockIdx.y*blockDim.y+threadIdx.y;
  if (ichan<nchan && iblock<nblock) {
    for (isum=0;isum<nsum;isum+=ndec) {
      idx2=ichan+nchan*(isum/ndec+iblock*nsum/ndec);
      for (idec=0,ztmp=0.0;idec<ndec;idec++) {
  idx1=ichan+nchan*(isum+idec+iblock*nsum);
  ztmp+=z[idx1];
      }
      ztmp/=(float) ndec;
      cz[idx2]=(float) ztmp;
    }
  }

  return;
}

int reshapeRawUdp(lofar_udp_reader *reader, int verbose) {

  if (lofar_udp_reader_step(reader) > 0) return 0;
  int nread = reader->meta->packetsPerIteration;
  if (verbose) {
    for (int i = 0; i < reader->meta->numPorts; i++) {
      if (reader->meta->portLastDroppedPackets[i] != 0) {
        printf("Port %d: %d dropped packets.\n", i, reader->meta->portLastDroppedPackets[i]);
      }
    }
  }
  nread *= UDPNTIMESLICE;

  return nread;
}
