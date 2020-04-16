# cdmt (forked)
CDMT, for *Coherent Dispersion Measure Trials*, is a software program to perform coherent dedispersion on complex voltage data from the LOFAR telescope and to coherently dedisperse to many different dispersion measure trials. It reads HDF5 input data and generates [SIGPROC](http://sigproc.sourceforge.net/) compatible filterbank files.

The software uses NVIDIA GPUs to accelerate the computations and hence requires compilation with `nvcc` against the `cufft` and `hdf5` libraries.

Presently the code is only capable of reading LOFAR HDF5 complex voltage data. If you want to use `cdmt` with a different input data type, let me know.


## Fork Details
This is a fork of Cees Bassa's CDMT; a GPU based coherent dedispersion implementation. It has been modified to take a set of 4 sigproc filterbank style files (S0.rawfil, S1.rawfil..., positive frequency order) with raw voltage data and a sigproc header as inputs to perform coherent dedispersion. There are also (currently untested; written while waiting for processing to finish) flags to limit reads to *n* time samples (*-r n*) and skip to *m* time samples in each input file (*-s m*).  

This modification was made for the observing setup used at the Irish LOFAR station, using Olaf Wucknitz's (MPIfRA) recording software and [udp2Filterbank](https://github.com/David-McKenna/udp2Filterbank) to record and preprocess data.


