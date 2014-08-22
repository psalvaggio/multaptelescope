// File Description
// Author: Philip Salvaggio

#ifndef ZERNIKE_CUDA_H_
#define ZERNIKE_CUDA_H_

#ifdef __CUDACC__

#include <cuda.h>

__global__ void zernike_kernel_4th(float* weights, float* output, int size);

#endif

#endif
