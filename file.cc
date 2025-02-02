/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <endian.h>
#include <stdlib.h>
#include <malloc.h>
#include <stdio.h>
#include <inttypes.h>

#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif

extern "C"
void inputData(char* fName, int* _numK, int* _numX,
               float** kx, float** ky, float** kz,
               float** x, float** y, float** z,
               float** phiR, float** phiI)
{
  int numK, numX;
  FILE* fid = fopen(fName, "r");

  if (fid == NULL)
    {
      fprintf(stderr, "Cannot open input file\n");
      exit(-1);
    }
  if(fread (&numK, sizeof (int), 1, fid) != 1 ) printf("error in file.cc fread!\n");
  *_numK = numK;
  if(fread (&numX, sizeof (int), 1, fid) != 1 ) printf("error in file.cc fread!\n");
  *_numX = numX;
  *kx = (float *) memalign(16, numK * sizeof (float));
  if(fread (*kx, sizeof (float), numK, fid) != numK ) printf("error in file.cc fread!\n");
  *ky = (float *) memalign(16, numK * sizeof (float));
  if(fread (*ky, sizeof (float), numK, fid) != numK ) printf("error in file.cc fread!\n");
  *kz = (float *) memalign(16, numK * sizeof (float));
  if(fread (*kz, sizeof (float), numK, fid) != numK ) printf("error in file.cc fread!\n");
  *x = (float *) memalign(16, numX * sizeof (float));
  if(fread (*x, sizeof (float), numX, fid) != numX ) printf("error in file.cc fread!\n");
  *y = (float *) memalign(16, numX * sizeof (float));
  if(fread (*y, sizeof (float), numX, fid) != numX ) printf("error in file.cc fread!\n");
  *z = (float *) memalign(16, numX * sizeof (float));
  if(fread (*z, sizeof (float), numX, fid) != numX ) printf("error in file.cc fread!\n");
  *phiR = (float *) memalign(16, numK * sizeof (float));
  if(fread (*phiR, sizeof (float), numK, fid) != numK ) printf("error in file.cc fread!\n");
  *phiI = (float *) memalign(16, numK * sizeof (float));
  if(fread (*phiI, sizeof (float), numK, fid) != numK ) printf("error in file.cc fread!\n");
  fclose (fid); 
}

extern "C"
void outputData(char* fName, float* outR, float* outI, int numX)
{
  FILE* fid = fopen(fName, "w");
  uint32_t tmp32;

  if (fid == NULL)
    {
      fprintf(stderr, "Cannot open output file\n");
      exit(-1);
    }

  /* Write the data size */
  tmp32 = numX;
  fwrite(&tmp32, sizeof(uint32_t), 1, fid);

  /* Write the reconstructed data */
  fwrite (outR, sizeof (float), numX, fid);
  fwrite (outI, sizeof (float), numX, fid);
  fclose (fid);
}
