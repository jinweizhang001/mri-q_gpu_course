/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include "support.h"
#include "support.cu"
#define PI   3.1415926535897932384626433832795029f
#define PIx2 6.2831853071795864769252867665590058f

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define K_ELEMS_PER_GRID 2048

struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};




__global__ void SampleAll(int M, float* rPhi, float* iPhi, float* __restrict__ phiMag) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  
  if(tid < M) {
    float real = rPhi[tid];
    float imag = iPhi[tid];
    phiMag[tid] = real*real + imag*imag;
  }
}

// inline
extern "C" void ComputePhiMag_GPU(int numK, float* phiR, float* phiI, 
                 float* __restrict__ phiMag) {
  int blk_num;
  const unsigned int blksize = 1024;
  blk_num = (numK - 1)/blksize + 1;
  float *A_d, *B_d, *C_d;
  Timer timer;
  // cudaError_t cuda_ret;
  
  startTime(&timer);
  // Allocate device variables ----------------------------------------------
  printf("Allocating device variables...\n"); fflush(stdout);
  cudaMalloc((void**) &A_d, sizeof(float)*numK   );
  cudaMalloc((void**) &B_d, sizeof(float)*numK   );
  cudaMalloc((void**) &C_d, sizeof(float)*numK   );
  cudaDeviceSynchronize();
  // Copy host variables to device
  printf("Copying data from host to device...\n"); fflush(stdout);
  cudaMemcpy(A_d, phiR, sizeof(float)*numK, cudaMemcpyHostToDevice   );
  cudaMemcpy(B_d, phiI, sizeof(float)*numK, cudaMemcpyHostToDevice   );
  cudaDeviceSynchronize();
  stopTime(&timer); printf("Coping data time: %f s\n", elapsedTime(timer));
  
  //int indexK = 0; // indexK is m, numK is number of samples, 2048 for 64
    //   for (indexK = 0; indexK < numK; indexK++) {
    //     float real = phiR[indexK];
    //     float imag = phiI[indexK];
    //     phiMag[indexK] = real*real + imag*imag;
    //   }

  // at each sample point m or indexK
  printf("Launching kernel...\n"); fflush(stdout);
  startTime(&timer);
  SampleAll <<<blk_num, blksize>>> (numK, A_d, B_d, C_d);
  cudaDeviceSynchronize();
  stopTime(&timer); printf("ComputePhiMag_GPU: %f s\n", elapsedTime(timer));
  
  // Copy device variables to host
  cudaMemcpy(phiMag, C_d, sizeof(float)*numK, cudaMemcpyDeviceToHost   );
  cudaDeviceSynchronize();

  // Free memory
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
  
  //stopTime(&timer); printf("ComputePhiMag_GPU: %f s\n", elapsedTime(timer));

}

