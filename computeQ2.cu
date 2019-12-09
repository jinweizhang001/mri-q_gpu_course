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
extern "C" void ComputePhiMagCPU(int numK, float* phiR, float* phiI, 
                 float* __restrict__ phiMag) {
  int blk_num;
  const unsigned int blksize = 1024;
  blk_num = (numK - 1)/blksize + 1;
  float *A_d, *B_d, *C_d;
  // cudaError_t cuda_ret;

  // Allocate device variables ----------------------------------------------
  printf("Allocating device variables..."); fflush(stdout);
  cudaMalloc((void**) &A_d, sizeof(float)*numK   );
  cudaMalloc((void**) &B_d, sizeof(float)*numK   );
  cudaMalloc((void**) &C_d, sizeof(float)*numK   );
  cudaDeviceSynchronize();
  // Copy host variables to device
  cudaMemcpy(A_d, phiR, sizeof(float)*numK, cudaMemcpyHostToDevice   );
  cudaMemcpy(B_d, phiI, sizeof(float)*numK, cudaMemcpyHostToDevice   );
  cudaDeviceSynchronize();
  //int indexK = 0; // indexK is m, numK is number of samples, 2048 for 64
    //   for (indexK = 0; indexK < numK; indexK++) {
    //     float real = phiR[indexK];
    //     float imag = phiI[indexK];
    //     phiMag[indexK] = real*real + imag*imag;
    //   }

  // at each sample point m or indexK
  SampleAll <<<blk_num, blksize>>> (numK, A_d, B_d, C_d);

  
  // cuda_ret = cudaDeviceSynchronize();
  // Copy device variables to host
  cudaMemcpy(phiMag, C_d, sizeof(float)*numK, cudaMemcpyDeviceToHost   );
  cudaDeviceSynchronize();

}

