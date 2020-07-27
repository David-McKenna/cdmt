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

#include "udpPacketManager/lofar_udp_reader.h"
#include "udpPacketManager/lofar_udp_misc.h"

#define HEADERSIZE 4096
#define DMCONSTANT 2.41e-10


// Struct for header information
struct header {
  int nchan,nbit=0,nsub,tel=11;
  double tstart,tsamp,fch1,foff,fcen,bwchan;
  double src_raj,src_dej;
  char source_name[80];
  char rawfname[4][1024];
};



struct header read_sigproc_header(char *fname, char *dataname, int ports);
void get_channel_chirp(double fcen,double bw,float dm,int nchan,int nbin,int nsub,cufftComplex *c);
__global__ void transpose_unpadd_and_detect(cufftComplex *cp1,cufftComplex *cp2,int nbin,int nchan,int nfft,int nsub,int noverlap,int nsamp,float *fbuf);
static __device__ __host__ inline cufftComplex ComplexScale(cufftComplex a,float s);
static __device__ __host__ inline cufftComplex ComplexMul(cufftComplex a,cufftComplex b);
static __global__ void PointwiseComplexMultiply(cufftComplex *a,cufftComplex *b,cufftComplex *c,int nx,int ny,int l,float scale);
__global__ void unpack_and_padd(char *dbuf0,char *dbuf1,char *dbuf2,char *dbuf3,int nsamp,int nbin,int nfft,int nsub,int noverlap,cufftComplex *cp1,cufftComplex *cp2);
__global__ void unpack_and_padd_first_iteration(char *dbuf0,char *dbuf1,char *dbuf2,char *dbuf3,int nsamp,int nbin,int nfft,int nsub,int noverlap,cufftComplex *cp1,cufftComplex *cp2);
__global__ void padd_next_iteration(char *dbuf0,char *dbuf1,char *dbuf2,char *dbuf3,int nsamp,int nbin,int nfft,int nsub,int noverlap,cufftComplex *cp1,cufftComplex *cp2);
__global__ void swap_spectrum_halves(cufftComplex *cp1,cufftComplex *cp2,int nx,int ny);
__global__ void compute_chirp(double fcen,double bw,float *dm,int nchan,int nbin,int nsub,int ndm,cufftComplex *c);
__global__ void compute_block_sums(float *z,int nchan,int nblock,int nsum,float *bs1,float *bs2);
__global__ void compute_channel_statistics(int nchan,int nblock,int nsum,float *bs1,float *bs2,float *zavg,float *zstd);
__global__ void redigitize(float *z,int nchan,int nblock,int nsum,float *zavg,float *zstd,float zmin,float zmax,unsigned char *cz);
__global__ void decimate_and_redigitize(float *z,int ndec,int nchan,int nblock,int nsum,float *zavg,float *zstd,float zmin,float zmax,unsigned char *cz);
__global__ void decimate(float *z,int ndec,int nchan,int nblock,int nsum,float *cz);
void write_to_disk_float(float* outputArray, FILE* outputFile, int nsamples, cudaEvent_t waitEvent);
void write_to_disk_char(unsigned char* outputArray, FILE* outputFile, int nsamples, cudaEvent_t waitEvent);
void write_filterbank_header(struct header h,FILE *file);
int reshapeRawUdp(lofar_udp_reader *reader);
long  __inline__ beamformed_packno(unsigned int timestamp, unsigned int sequence);

extern "C"  {
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
  printf("-s <packets>       Number of packets to skip in the filterbank before stating processing [integer, default: 0]\n");
  printf("-r <packets>       Number of packets to read in total from the -s offset [integer, default: length of file]\n");
  printf("-m <sigproc header location>  Sigproc header to read metadata from [default: fil prefix.sigprochdr]\n");
  printf("-f <FFTs per op> Number of FFTs to execute per cuFFT call [default: 128]\n");
  printf("-a               Disable redigitisation; output float32 [default: false]\n");
  printf("-c <num chan>    Channelisation Factor [default: 8]\n");

  return;
}

int main(int argc,char *argv[])
{
  int i,j,nsamp,nfft,mbin,nvalid,nchan=8,nbin=65536,noverlap=2048,nsub=20,ndm,ndec=1;
  int idm,iblock,nread_tmp,nread,mchan,msamp,mblock,msum=1024;
  char *header,*udpbuf[4],*dudpbuf[4];
  FILE *file;
  unsigned char **cbuf[2],*dcbuf;
  float **cbuff[2], *dcbuff;
  float *fbuf,*dfbuf;
  float *bs1,*bs2,*zavg,*zstd;
  cufftComplex *cp1,*cp2,*dc,*cp1p,*cp2p;
  cufftHandle ftc2cf,ftc2cb;
  int idist,odist,iembed,oembed,istride,ostride;
  dim3 blocksize,gridsize;
  struct header hdr;
  clock_t startclock;
  float *dm,*ddm,dm_start,dm_step;
  char fname[128],fheader[1024],*udpfname,sphdrfname[1024] = "",obsid[128]="cdmt";
  int bytes_read;
  long int ts_read=LONG_MAX,ts_skip=0;
  long int total_ts_read=0;
  int part=0,device=0,verbose=0,nforward=128,redig=1,ports=4;
  int arg=0;
  FILE **outfile;

  lofar_udp_reader *reader;

  // Read options
  if (argc>1) {
    while ((arg=getopt(argc,argv,"ac:p:f:d:D:ho:b:N:n:s:r:m:t:"))!=-1) {
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
  ts_skip=atol(optarg) / 16;
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

      case 'h':
  usage();
  return 0;

  return 0;
      }
    }
  } else {
    printf("Unknown option '%c'\n", arg);
    usage();
    return 0;
  }
  udpfname=argv[optind];

  // Sanity checks to avoid voids in output filterbank
  if (nbin % 8 != 0) {
    fprintf(stderr, "ERROR: nbin must be disible by 8 (currently %d, remainder %d). Exiting.\n", nbin, nbin % 8);
    exit(1);
  }
  if ( (nforward * (nbin-2*noverlap)) % 8 != 0 ) {
    fprintf(stderr, "ERROR: Valid data length must be divisible by 8 (currently %d, remainer %d). Exiting.", nbin-2*noverlap, (nbin-2*noverlap) % 8);
    exit(1);
  }

  if ((nforward * (nbin-2*noverlap) / 8) % 1024 != 0) {
    fprintf(stderr, "ERROR: Interal sum cannot proceed; valid samples must be divisible by 1024 (currently %d, remainder %d).\n", (nforward * (nbin-2*noverlap) / 8), (nforward * (nbin-2*noverlap) / 8) % 1024);
    fprintf(stderr, "Consider using %d or %d as your forward FFT size next time. Exiting.\n", 64 * ((nforward * (nbin-2*noverlap) / 8) - (nforward * (nbin-2*noverlap) / 8) % 1024) / 1024 + 2 * noverlap,
                                                                                   64 * ((nforward * (nbin-2*noverlap) / 8) + (1024  - (nforward * (nbin-2*noverlap) / 8) % 1024)) / 1024 + 2 * noverlap);
    exit(1);
  }
  

  if (strcmp(sphdrfname, "") == 0) {
    sprintf(sphdrfname, "%s.sigprochdr", udpfname);
  }

  FILE* inputFiles[4];
  int compressedInput = 0;
  char tmpfname[1024] = "";
  if (strstr(udpfname, "zst") != NULL) compressedInput = 1;


 
  // Read sigproc header
  hdr = read_sigproc_header(sphdrfname, udpfname, ports);
  const double stg1 = (1.0 / 2.41e-4) *  abs(pow((double) hdr.fch1 + hdr.nsub * hdr.foff + hdr.foff *0.5,-2.0) - pow((double) hdr.fch1 + hdr.nsub * hdr.foff - hdr.foff *0.5, -2.0));
  const int overlapCheck = (int) (stg1 * (dm_start + dm_step * ndm)/ hdr.tsamp);
  if (overlapCheck > nbin) {
    fprintf(stderr, "WARNING: The size of your FFT bin is too short for the given DMs and frequencies. Given bin size: %d, Suggested minimum bin size: %d\n", nbin, overlapCheck);
  } else if (overlapCheck > noverlap) {
    fprintf(stderr, "WARNING: The size of your FFT overlap is too short for the given maximum DM. Given overlap: %d, Suggested minimum overlap: %d.\n", noverlap, overlapCheck);
  }


  for (int i = 0; i < 4; i++) {
    sprintf(tmpfname, udpfname, i);
    printf("Opening %s...\n", tmpfname);

    inputFiles[i] = fopen(tmpfname, "r");

    if (inputFiles[i] == NULL) {
      printf("Input file failed to open (null pointer)\n");
    }

    strcpy(hdr.rawfname[i],tmpfname);
  }
  // Read the number of subbands
  nsub=hdr.nsub;
  double timeOffset = hdr.tsamp;

  // Adjust header for filterbank format
  hdr.tsamp*=nchan*ndec;
  hdr.nchan=nsub*nchan;
  if (redig) hdr.nbit=8;
  else hdr.nbit=32;
  hdr.fch1=hdr.fcen+0.5*hdr.nsub*hdr.bwchan-0.5*hdr.bwchan/nchan;
  hdr.foff=-fabs(hdr.bwchan/nchan);


  // Data size
  nvalid=nbin-2*noverlap;
  nsamp=nforward*nvalid;
  nfft=(int) ceil(nsamp/(float) nvalid);
  mbin=nbin/nchan; // nbin must be evenly divisible by 8
  mchan=nsub*nchan;
  msamp=nsamp/nchan; // nforward * nvalid must be divisble by 8
  mblock=msamp/msum; // nforward * nvalid / 8 must be disible by 1024

  long unsigned int bytesUsed = sizeof(cufftComplex) * nbin * nfft * nsub * 4 + sizeof(cufftComplex) * nbin *nsub * ndm + sizeof(float) * mblock * mchan * 2 + sizeof(char) * nsamp * nsub * 4 + sizeof(float) * nsamp * nsub + redig * msamp * mchan / ndec - (redig - 1) * 4 * msamp * mchan * ndec;
  printf("We anticipate %ld GB (%ld bytes) to be allocated on the GPU.\n", bytesUsed >> 30, bytesUsed);


  const long int packetGulp = nsamp / 16;
  reader = lofar_udp_meta_file_reader_setup(inputFiles, ports, 1, 11, 1, packetGulp, (long) -1, LONG_MAX, compressedInput);
  if (reader == NULL) {
    fprintf(stderr, "Failed to generate LOFAR UDP Reader, exiting.\n");
    exit(1);
  }

  if (ts_skip > 0) {
      if (int returnVal = lofar_udp_file_reader_reuse(reader, reader->meta->lastPacket + ts_skip, -1) > 0) {
        fprintf(stderr, "Error re-initialising reader for %ld packets skipped (error %d), exiting.\n", ts_skip, returnVal);
        exit(1);
      }

      printf("Skipped %ld time stemps.\n", ts_skip * 16 );
  }

  hdr.tstart = lofar_get_packet_time_mjd(reader->meta->inputData[0]);
  if (verbose) printf("lofar_udp_reader Generated successfully.\n");

  // Set device
  checkCudaErrors(cudaSetDevice(device));

  // Generate streams for asyncrnous operations
  cudaStream_t streams[3];
  for (i = 0; i < 3; i++)
    checkCudaErrors(cudaStreamCreate(&(streams[i])));

  // Create 2 events; one which blocks execution (preventing new data reads) and the other waiting for compute to finish.
  cudaEvent_t events[3];
  cudaEventCreateWithFlags(&(events[0]), cudaEventBlockingSync & cudaEventDisableTiming);
  cudaEventCreateWithFlags(&(events[1]), cudaEventDisableTiming);
  cudaEventCreateWithFlags(&(events[2]), cudaEventDisableTiming);

  cudaEvent_t dmWriteEvents[2][ndm];
  for (i =0; i < ndm; i++)
    for (j = 0; j < 2; j++)
      cudaEventCreateWithFlags(&(dmWriteEvents[j][i]), cudaEventBlockingSync & cudaEventDisableTiming);

  // DMcK: cuFFT docs say it's best practice to plan before allocating memory
  // cuda-memcheck fails initialisation before this block is run?
  // Generate FFT plan (batch in-place forward FFT)
  idist=nbin;  odist=nbin;  iembed=nbin;  oembed=nbin;  istride=1;  ostride=1;
  checkCudaErrors(cufftPlanMany(&ftc2cf,1,&nbin,&iembed,istride,idist,&oembed,ostride,odist,CUFFT_C2C,nfft*nsub));
  cufftSetStream(ftc2cf,streams[0]);
  cudaDeviceSynchronize();
  // Total malloc (FFT forward)

  // Generate FFT plan (batch in-place backward FFT)
  idist=mbin;  odist=mbin;  iembed=mbin;  oembed=mbin;  istride=1;  ostride=1;
  checkCudaErrors(cufftPlanMany(&ftc2cb,1,&mbin,&iembed,istride,idist,&oembed,ostride,odist,CUFFT_C2C,nchan*nfft*nsub));
  cufftSetStream(ftc2cb,streams[0]);
  cudaDeviceSynchronize();
  // Total malloc (backward)

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

  // Allocate memory for redigitized output and header
  header=(char *) malloc(sizeof(char)*HEADERSIZE);
  for (i=0;i<4;i++) {
    udpbuf[i]= reader->meta->outputData[i];
    checkCudaErrors(cudaMalloc((void **) &dudpbuf[i], (size_t) sizeof(char)*nsamp*nsub));
  }

  // Allocate output buffers
  fbuf=(float *) malloc(sizeof(float)*nsamp*nsub);
  checkCudaErrors(cudaMalloc((void **) &dfbuf, (size_t) sizeof(float)*nsamp*nsub));
  
  if (redig) {
    for (i = 0; i < ndm; i++)
      for (j = 0; j < 2; j++)
        cbuf[j][i]=(unsigned char *) malloc(sizeof(unsigned char)*msamp*mchan/ndec);
    checkCudaErrors(cudaMalloc((void **) &dcbuf, (size_t) sizeof(unsigned char)*msamp*mchan/ndec));
  } else {
    for (i = 0; i < ndm; i++)
      for (j = 0; j < 2; j++)
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
  if (verbose) printf("Malloc complete.\n");
  // Compute chirp
  blocksize.x=32; blocksize.y=32; blocksize.z=1;
  gridsize.x=nsub/blocksize.x+1; gridsize.y=nchan/blocksize.y+1; gridsize.z=ndm/blocksize.z+1;
  compute_chirp<<<gridsize,blocksize>>>(hdr.fcen,nsub*hdr.bwchan,ddm,nchan,nbin,nsub,ndm,dc);
  if (verbose) printf("Chirp calculated.\n");
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
  if (verbose) printf("Output header generated successfully.\n");
  // Write headers
  for (idm=0;idm<ndm;idm++) {
    // Send header
    fwrite(fheader,sizeof(char),bytes_read,outfile[idm]);
  }
  if (verbose) printf("Output header written successfully.\n");
  if (verbose) printf("Starting processing loop.\n\n");

  // Loop over input file contents
  double timeInSeconds = 0.0;
  nread = INT_MAX;

  omp_lock_t readingLock;
  omp_init_lock(&readingLock);

  // Skip the first noverlap samples as they are 0'd
  int writeOffset = noverlap * 2;
  #pragma omp parallel for ordered schedule(static, 1) private(writeOffset)
  for (iblock=0;;iblock++) {
    #pragma omp cancellation point parallel

    if (iblock != 0)
      writeOffset = 0;
    else
      writeOffset = noverlap * 2;

    // Ge tthe current stream from block iteration
    int streamIdx = iblock % 2;
    cudaStream_t stream = streams[streamIdx];
    // Read block
    #pragma omp ordered single
    {

      cudaEventSynchronize(events[0]);
      startclock=clock();
      nread_tmp = reshapeRawUdp(reader);

      if (nread > nread_tmp) {
        nread = nread_tmp;
      }

      if (nread==0) {
        printf("No data read from last file; assuming EOF, finishng up.\n");
        #pragma omp cancel parallel for
      } else if (iblock != 0 && nread < nread_tmp) {
        printf("Received less data than expected; we may have parsed out of order data or we are nearing the EOF.\n");
      }

      // Count up the total bytes read
      total_ts_read += nread;
      printf("Block: %d: Read %ld MB in %.2f s\n",iblock,sizeof(char)*nread*nsub*4/(1<<20),(float) (clock()-startclock)/CLOCKS_PER_SEC);

      omp_set_lock(&readingLock);
      cudaStreamWaitEvent(stream, events[1], 0);
      // Copy buffers to device
      startclock=clock();
      for (i=0;i<4;i++)
        checkCudaErrors(cudaMemcpyAsync(dudpbuf[i],udpbuf[i],sizeof(char)*nread*nsub,cudaMemcpyHostToDevice,stream));

      checkCudaErrors(cudaEventRecord(events[0], stream));
    }
    // Unpack data and padd data
    blocksize.x=32; blocksize.y=32; blocksize.z=1;
    gridsize.x=nbin/blocksize.x+1; gridsize.y=nfft/blocksize.y+1; gridsize.z=nsub/blocksize.z+1;
    if (iblock > 0) {
      unpack_and_padd<<<gridsize,blocksize,0,stream>>>(dudpbuf[0],dudpbuf[1],dudpbuf[2],dudpbuf[3],nread,nbin,nfft,nsub,noverlap,cp1p,cp2p);
    }
    else {
      unpack_and_padd_first_iteration<<<gridsize,blocksize,0,stream>>>(dudpbuf[0],dudpbuf[1],dudpbuf[2],dudpbuf[3],nread,nbin,nfft,nsub,noverlap,cp1p,cp2p);;
    }

    // Perform FFTs
    cufftSetStream(ftc2cf, stream);
    cufftSetStream(ftc2cb, stream);
    checkCudaErrors(cufftExecC2C(ftc2cf,(cufftComplex *) cp1p,(cufftComplex *) cp1p,CUFFT_FORWARD));
    checkCudaErrors(cufftExecC2C(ftc2cf,(cufftComplex *) cp2p,(cufftComplex *) cp2p,CUFFT_FORWARD));

    // Swap spectrum halves for large FFTs
    blocksize.x=32; blocksize.y=32; blocksize.z=1;
    gridsize.x=nbin/blocksize.x+1; gridsize.y=nfft*nsub/blocksize.y+1; gridsize.z=1;
    swap_spectrum_halves<<<gridsize,blocksize,0,stream>>>(cp1p,cp2p,nbin,nfft*nsub);

    // Loop over dms
    for (idm=0;idm<ndm;idm++) {

      // Perform complex multiplication of FFT'ed data with chirp
      blocksize.x=32; blocksize.y=32; blocksize.z=1;
      gridsize.x=nbin*nsub/blocksize.x+1; gridsize.y=nfft/blocksize.y+1; gridsize.z=1;
      PointwiseComplexMultiply<<<gridsize,blocksize,0,stream>>>(cp1p,dc,cp1,nbin*nsub,nfft,idm,1.0/(float) nbin);
      PointwiseComplexMultiply<<<gridsize,blocksize,0,stream>>>(cp2p,dc,cp2,nbin*nsub,nfft,idm,1.0/(float) nbin);
      
      // end of cp1p/cp2p needed

      if (idm == ndm - 1) {
        checkCudaErrors(cudaEventRecord(events[2], stream));
        cudaStreamWaitEvent(streams[2], events[2], 0);

        blocksize.x=32; blocksize.y=32; blocksize.z=1;
        gridsize.x=nbin/blocksize.x+1; gridsize.y=nfft/blocksize.y+1; gridsize.z=nsub/blocksize.z+1;
        padd_next_iteration<<<gridsize,blocksize,0,streams[2]>>>(dudpbuf[0],dudpbuf[1],dudpbuf[2],dudpbuf[3],nread,nbin,nfft,nsub,noverlap,cp1p,cp2p);
        checkCudaErrors(cudaEventRecord(events[1], streams[2]));
        omp_unset_lock(&readingLock);
      }
      // Swap spectrum halves for small FFTs
      blocksize.x=32; blocksize.y=32; blocksize.z=1;
      gridsize.x=mbin/blocksize.x+1; gridsize.y=nchan*nfft*nsub/blocksize.y+1; gridsize.z=1;
      swap_spectrum_halves<<<gridsize,blocksize,0,stream>>>(cp1,cp2,mbin,nchan*nfft*nsub);
      
      // Perform FFTs
      checkCudaErrors(cufftExecC2C(ftc2cb,(cufftComplex *) cp1,(cufftComplex *) cp1,CUFFT_INVERSE));
      checkCudaErrors(cufftExecC2C(ftc2cb,(cufftComplex *) cp2,(cufftComplex *) cp2,CUFFT_INVERSE));
      
      cudaStreamWaitEvent(stream, dmWriteEvents[streamIdx][idm-1 > -1 ? idm-1 : 0], 0);
      // Detect data
      blocksize.x=32; blocksize.y=32; blocksize.z=1;
      gridsize.x=mbin/blocksize.x+1; gridsize.y=nchan/blocksize.y+1; gridsize.z=nfft/blocksize.z+1;
      transpose_unpadd_and_detect<<<gridsize,blocksize,0,stream>>>(cp1,cp2,mbin,nchan,nfft,nsub,noverlap/nchan,nread/nchan,dfbuf);
      
      // end of cp1/cp2 needed
      
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

        cudaEventRecord(dmWriteEvents[streamIdx][idm], stream);

        #pragma omp task
        {
          write_to_disk_char(&(cbuf[streamIdx][idm][writeOffset*nsub/ndec]), outfile[idm], (nread-writeOffset)*nsub/ndec, dmWriteEvents[streamIdx][idm]);
        }

      } else {
        if (ndec==1) {
          checkCudaErrors(cudaMemcpyAsync(cbuff[streamIdx][idm], dfbuf,sizeof(float)*msamp*mchan,cudaMemcpyDeviceToHost,stream));
          
        } else {
          blocksize.x=32; blocksize.y=32; blocksize.z=1;
          gridsize.x=mchan/blocksize.x+1; gridsize.y=mblock/blocksize.y+1; gridsize.z=1;
          decimate<<<gridsize,blocksize,0,stream>>>(dfbuf,ndec,mchan,mblock,msum,dcbuff);
          checkCudaErrors(cudaMemcpyAsync(cbuff[streamIdx][idm],dcbuff,sizeof(float)*msamp*mchan/ndec,cudaMemcpyDeviceToHost,stream));
        }

        cudaEventRecord(dmWriteEvents[streamIdx][idm], stream);

        #pragma omp task
        {
          write_to_disk_float(&(cbuff[streamIdx][idm][writeOffset*nsub/ndec]), outfile[idm], (nread-writeOffset)*nsub/ndec, dmWriteEvents[streamIdx][idm]);
        }
      }
    }

    #pragma omp taskwait

    printf("Processed %d DMs in %.2f s\n",ndm,(float) (clock()-startclock)/CLOCKS_PER_SEC);
    timeInSeconds += (double) (nread - writeOffset) * timeOffset;
    printf("Current data processed: %02ld:%02ld:%05.2lf (%1.2lfs)\n\n", (long int) (timeInSeconds / 3600.0), (long int) ((fmod(timeInSeconds, 3600.0)) / 60.0), fmod(timeInSeconds, 60.0), timeInSeconds);
    // Exit when we pass the read length limit
    if (total_ts_read > ts_read) {
      #pragma omp cancel parallel for
    }

  }

  omp_destroy_lock(&readingLock);
  // Close files
  for (i=0;i<ndm;i++)
    fclose(outfile[i]);

  // Reader cleanup
  lofar_udp_reader_cleanup(reader);

  // Free
  free(header);
  for (i=0;i<4;i++) {
    cudaFree(dudpbuf);  
  }
  free(fbuf);
  free(dm);
  free(outfile);

  if (redig) {
    for (i = 0; i < ndm; i++)
      for (j =0; j < 2; j++)
        free(cbuf[j][i]);
    cudaFree(bs1);
    cudaFree(bs2);
    cudaFree(zavg);
    cudaFree(zstd);
    cudaFree(dcbuf);
  } else {
    for (i = 0; i < ndm; i++)
      for (j =0; j < 2; j++)
        free(cbuff[j][i]);
    if (ndec > 1) cudaFree(dcbuff);
  }

  cudaFree(dfbuf);
  cudaFree(cp1);
  cudaFree(cp2);
  cudaFree(cp1p);
  cudaFree(cp2p);
  cudaFree(dc);
  cudaFree(ddm);

  // Free plan
  cufftDestroy(ftc2cf);
  cufftDestroy(ftc2cb);

  for(i = 0; i < 3; i++)
    cudaStreamDestroy(streams[i]);
  for(i = 0; i < 2; i++)
    cudaEventDestroy(events[i]);

  for (i = 0; i < ndm; i++)
    for (j =0; j < 2; j++)
      cudaEventDestroy(dmWriteEvents[j][i]);

  return 0;
}


void inline write_to_disk_float(float* outputArray, FILE* outputFile, int nsamples, cudaEvent_t waitEvent)
{
  cudaEventSynchronize(waitEvent);
  fwrite(outputArray,sizeof(float),nsamples, outputFile); 
}

void inline write_to_disk_char(unsigned char* outputArray, FILE* outputFile, int nsamples, cudaEvent_t waitEvent)
{
  cudaEventSynchronize(waitEvent);
  fwrite(outputArray,sizeof(char),nsamples, outputFile); 
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
      fseek(inputfile, sizeof(int), SEEK_CUR);
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
__global__ void unpack_and_padd(char *dbuf0,char *dbuf1,char *dbuf2,char *dbuf3,int nsamp,int nbin,int nfft,int nsub,int noverlap,cufftComplex *cp1,cufftComplex *cp2)
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
__global__ void unpack_and_padd_first_iteration(char *dbuf0,char *dbuf1,char *dbuf2,char *dbuf3,int nsamp,int nbin,int nfft,int nsub,int noverlap,cufftComplex *cp1,cufftComplex *cp2)
{
  int64_t ibin,ifft,isamp,isub,idx1,idx2;

  // Indices of input data
  ibin=blockIdx.x*blockDim.x+threadIdx.x;
  ifft=blockIdx.y*blockDim.y+threadIdx.y;
  isub=blockIdx.z*blockDim.z+threadIdx.z;

  // Only compute valid threads
  if (ibin<nbin && ifft<nfft && isub<nsub) {
    isamp=ibin+(nbin-2*noverlap)*ifft-noverlap;
    if (isamp >= 2*noverlap) {
      idx1=ibin+nbin*isub+nsub*nbin*ifft;

      idx2=isub+nsub*(isamp-2*noverlap);
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
__global__ void padd_next_iteration(char *dbuf0,char *dbuf1,char *dbuf2,char *dbuf3,int nsamp,int nbin,int nfft,int nsub,int noverlap,cufftComplex *cp1,cufftComplex *cp2)
{
  int64_t ibin,ifft,isamp,isub,idx1,idx2;

  // Indices of input data
  ibin=blockIdx.x*blockDim.x+threadIdx.x;
  ifft=blockIdx.y*blockDim.y+threadIdx.y;
  isub=blockIdx.z*blockDim.z+threadIdx.z;

  // Only compute valid threads
  if (ibin<nbin && ifft<nfft && isub<nsub) {
    isamp=ibin+(nbin-2*noverlap)*ifft-noverlap;
    if (isamp<noverlap) {
      // VVV FIX
      idx1=ibin+nbin*isub+nsub*nbin*ifft;
      idx2=isub+nsub*(isamp+nsamp-noverlap);
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
  send_int("machine_id",11,file);
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

int reshapeRawUdp(lofar_udp_reader *reader) {

  if (lofar_udp_reader_step(reader) > 0) return 0;
  int nread = reader->meta->packetsPerIteration;

 // MODE 11

  nread *= 16;

  return nread;
}
