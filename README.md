# matrix_lu
matrix_lu is a simple program that calculates LU-decomposition of a large matrix and
verifies the results.  This program is implemented using two ways: 
    1. Data Parallel C++ (DPC++)
    2. OpenMP (omp)

For comprehensive instructions regarding DPC++ Programming, go to https://software.intel.com/en-us/oneapi-programming-guide and search based on relevant terms noted in the comments.
  
| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04, Windows 10*
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler beta, Intel&reg; C/C++ Compiler beta
| What you will learn               | Offloads computations on 2D arrays to GPU using Intel DPC++ and OpenMP
| Time to complete                  | 15 minutes  

### Purpose
The code will attempt to run the calculation
on both the GPU and CPU, and then verifies the results. The size of the
computation can be adjusted for heavier workloads (defined below). If
successful, the name of the offload device and a success message are
displayed.

This sample uses buffers to manage memory.

matrix_lu includes C++ implementations of both Data Parallel (DPC++) and
OpenMP; each is contained in its own .cpp file. This provides a way to compare
existing offload techniques such as OpenMP with Data Parallel C++ within a
relatively simple sample. The default will build the DPC++ application.
Separate OpenMP build instructions are provided below. Note: matrix_lu does not
support OpenMP on Windows.

The code will attempt first to execute on an available GPU and fallback to the system's CPU if a compatible GPU is not detected.  The device used for compilation is displayed in the output.

## Key implementation details
SYCL implementation explained.
OpenMP offload implementation explained.

## License  
This code sample is licensed under MIT license. 

### How to build for DPC++ on Linux  
   * Build the program using Make  
    cd matrix_lu &&  
    make all  

   * Run the program  
    make run  

   * Clean the program  
    make clean 

### How to Build for OpenMP on Linux  
   * Build the program using Make  
    cd matrix_lu &&  
    make build_omp  

   * Run the program  
    make run_omp  

   * Clean the program  
    make clean

### How to build for DPC++ on Windows
The OpenMP offload target is not supported on Windows yet.

#### Command Line using nmake
   Build matrix_lu DPCPP version
   * nmake -f Makefile.win build_dpcpp
   * nmake -f Makefile.win run_dpcpp  

### How to build for OpenMP on Windows
The OpenMP offload target is not supported on Windows at this time.

## Running the Sample

### Application Parameters 
You can modify the size of the computation by adjusting the size parameter in the dpcpp and omp .cpp files. The configurable parameters include:
   size = N = 2500;

