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

__constant__ struct kValues kVals_c[3072];


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

__global__ void cmpQ(int numK, int numX, 
  float* gx, float* gy, float* gz, 
  float *__restrict__ Qr, float *__restrict__ Qi) {

  // __shared__ float ds_kVals[sizeof(kVals)];

  float expArg;
  float cosArg;
  float sinArg;
  // find the index of voxel assigned to this thread
  //threadIdx.x + blockDim.x * blockIdx.x;
  int n = blockIdx.x * blockDim.x + threadIdx.x;

  // register allocate voxel inputs and outputs
  if(n < numX) {
    float x = gx[n];
    float y = gy[n];
    float z = gz[n];
    float Qracc = 0.0f;
    float Qiacc = 0.0f;
    // m is indexK
    for(int m = 0; m < numK; m++) {
      // better to store sample data kVals[] in constant memory
      expArg = PIx2 * (kVals_c[m].Kx * x +
                    kVals_c[m].Ky * y +
                    kVals_c[m].Kz * z);

      cosArg = cosf(expArg);
      sinArg = sinf(expArg);
      float phi = kVals_c[m].PhiMag;
      Qracc += phi * cosArg;
      Qiacc += phi * sinArg;

    }
    __syncthreads();
    Qr[n] = Qracc;
    Qi[n] = Qiacc;
  }
}

// __global__ void cmpQ(int numK, int numX, struct kValues *kVals, 
//   float* gx, float* gy, float* gz, 
//   float *__restrict__ Qr, float *__restrict__ Qi) {

//   // __shared__ float ds_kVals[sizeof(kVals)];

//   float expArg;
//   float cosArg;
//   float sinArg;
//   // find the index of voxel assigned to this thread
//   //threadIdx.x + blockDim.x * blockIdx.x;
//   int n = blockIdx.x * blockDim.x + threadIdx.x;

//   // register allocate voxel inputs and outputs
//   if(n < numX) {
//     float x = gx[n];
//     float y = gy[n];
//     float z = gz[n];
//     float Qracc = 0.0f;
//     float Qiacc = 0.0f;
//     // m is indexK
//     for(int m = 0; m < numK; m++) {
//       // better to store sample data kVals[] in constant memory
//       expArg = PIx2 * (kVals[m].Kx * x +
//                     kVals[m].Ky * y +
//                     kVals[m].Kz * z);

//       cosArg = cosf(expArg);
//       sinArg = sinf(expArg);
//       float phi = kVals[m].PhiMag;
//       Qracc += phi * cosArg;
//       Qiacc += phi * sinArg;

//     }
//     __syncthreads();
//     Qr[n] = Qracc;
//     Qi[n] = Qiacc;
//   }
// }

extern "C" void ComputeQ_GPU(int numK, int numX, struct kValues *kVals, 
  float* x, float* y, float* z, 
  float *__restrict__ Qr, float *__restrict__ Qi) {

  int blk_num;
  const unsigned int blksize = 1024;
  blk_num = (numX - 1)/blksize + 1;
  float *x_d, *y_d, *z_d;
  float *__restrict__ Qr_d; 
  float *__restrict__ Qi_d;
  // struct kValues *kVals_d;
  Timer timer;

  startTime(&timer);
  // Allocate device variables ----------------------------------------------
  printf("Allocating device variables...\n"); fflush(stdout);
  cudaMalloc((void**) &x_d, sizeof(float)*numX   );
  cudaMalloc((void**) &y_d, sizeof(float)*numX   );
  cudaMalloc((void**) &z_d, sizeof(float)*numX   );
  // cudaMalloc((void**) &kVals_d, sizeof(struct kValues)*numK   );
  cudaMalloc((void**) &Qr_d, sizeof(float)*numX   );
  cudaMalloc((void**) &Qi_d, sizeof(float)*numX   );

  cudaDeviceSynchronize();
  // Copy host variables to device
  printf("Copying data from host to device...\n"); fflush(stdout);
  cudaMemcpy(x_d, x, sizeof(float)*numX, cudaMemcpyHostToDevice   );
  cudaMemcpy(y_d, y, sizeof(float)*numX, cudaMemcpyHostToDevice   );
  cudaMemcpy(z_d, z, sizeof(float)*numX, cudaMemcpyHostToDevice   );
  // cudaMemcpy(kVals_d, kVals, sizeof(struct kValues)*numK, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(kVals_c, kVals, sizeof(struct kValues)*numK, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  stopTime(&timer); printf("Coping data to device time: %f s\n", elapsedTime(timer)); 

  // Launch a kernel
  printf("Launching kernel...\n"); fflush(stdout);
  startTime(&timer);
  // cmpQ <<<blk_num, blksize>>> (numK, numX, kVals_d, x_d, y_d, z_d, Qr_d, Qi_d);
  cmpQ <<<blk_num, blksize>>> (numK, numX, x_d, y_d, z_d, Qr_d, Qi_d);
  cudaDeviceSynchronize();
  stopTime(&timer); printf("ComputeQ_GPU kernel time: %f s\n", elapsedTime(timer));  

  // Copy device variables to host
  startTime(&timer);
  cudaMemcpy(Qr, Qr_d, sizeof(float)*numX, cudaMemcpyDeviceToHost   );
  cudaMemcpy(Qi, Qi_d, sizeof(float)*numX, cudaMemcpyDeviceToHost   );
  cudaDeviceSynchronize();

  // Free memory
  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);
  // cudaFree(kVals_d);
  cudaFree(kVals_c);
  cudaFree(Qr_d);
  cudaFree(Qi_d);
  stopTime(&timer); printf("Copying data back time: %f s\n", elapsedTime(timer));  

}
