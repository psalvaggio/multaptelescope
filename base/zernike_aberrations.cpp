// File Description
// Author: Philip Salvaggio

#include "zernike_aberrations.h"
#include "zernike_cuda.h"

#include <cmath>
#include <iostream>

#ifdef __CUDACC__
#include <cuda.h>
#endif

using namespace cv;
using std::vector;

ZernikeAberrations::ZernikeAberrations() 
    : gpu_weights_(NULL), gpu_wfe_(NULL), gpu_wfe_size_(0) {
  #ifdef __CUDACC__
  cudaMalloc(&gpu_weights_, 9 * sizeof(float));
  #endif
}

ZernikeAberrations::~ZernikeAberrations() {
  #ifdef __CUDACC__
  if (gpu_weights_) cudaFree(gpu_weights_);
  if (gpu_wfe_) cudaFree(gpu_wfe_);
  #endif
}

void ZernikeAberrations::aberrations(const vector<double>& weights,
                                     size_t output_size,
                                     Mat* output) {
  if (!output) return;

  const size_t kSize = output_size * output_size;
  
  #ifdef __CUDACC__

  const int kBlockSize = 1024;

  int num_blocks = (kSize % kBlockSize == 0)
      ? kSize / kBlockSize : kSize / kBlockSize + 1;

  float cpu_weights[9];
  for (int i = 0; i < 9; i++) {
    cpu_weights[i] = (i < weights.size()) ? weights[i] : 0;
  }

  if (gpu_weights_ == NULL) {
    cudaMalloc(&gpu_weights_, 9 * sizeof(float));
  }
  cudaMemcpy(gpu_weights_, cpu_weights, 9 * sizeof(float),
      cudaMemcpyHostToDevice);

  if (output_size != gpu_wfe_size_ && gpu_wfe_) {
    cudaFree(gpu_wfe_);
    gpu_wfe_ = NULL;
  }
  if (gpu_wfe_ == NULL) {
    cudaMalloc(&gpu_wfe_, kSize * sizeof(float));
  }

  dim3 block, grid;
  block.x = kBlockSize; block.y = 1; block.z = 1;
  grid.x = num_blocks; grid.y = 1; grid.z = 1;
  zernike_kernel_4th<<<grid, block>>>(gpu_weights_, gpu_wfe_, output_size);

  output->create(output_size, output_size, CV_32FC1);
  cudaMemcpy(output->data, gpu_wfe_, kSize * sizeof(float),
      cudaMemcpyDeviceToHost);
  output->convertTo(*output, CV_64F);

  #else

  const double kCenter = 0.5 * (output_size - 1);

  output->create(output_size, output_size, CV_64F);
  *output = Scalar(0);

  double* output_data = reinterpret_cast<double*>(output->data);

  for (size_t i = 0; i < kSize; i++) {
    double x = (i % output_size) - kCenter;
    double y = (i / output_size) - kCenter;
    double rho = sqrt(x*x + y*y) / kCenter;

    if (rho > 1) {
      continue;
    }

    double theta = atan2(y, x);
    double cos_theta = cos(theta);
    double sin_theta = sin(theta);

    double wfe = 0;
    for (size_t j = 0; j < weights.size(); j++) {
      switch(j) {
        case 0:
          wfe += weights[0]; break;
        case 1:
          wfe += weights[1] * rho * cos_theta; break;
        case 2:
          wfe += weights[2] * rho * sin_theta; break;
        case 3:
          wfe += weights[3] * (rho*rho - 1); break;
        case 4:
          wfe += weights[4] * rho * rho * cos(2 * theta); break;
        case 5:
          wfe += weights[5] * rho * rho * sin(2 * theta); break;
        case 6:
          wfe += weights[6] * rho * (3 * rho * rho - 2) * cos_theta; break;
        case 7:
          wfe += weights[7] * rho * (3 * rho * rho - 2) * sin_theta; break;
        case 8:
          wfe += weights[8] * (1 - 6 * rho * rho + 6 * pow(rho, 4)); break;
        case 9:
          wfe += weights[9] * rho * rho * rho * cos(3 * theta); break;
        case 10:
          wfe += weights[10] * rho * rho * rho * sin(3 * theta); break;
        case 11:
          wfe += weights[11] * rho * rho * (4 * rho * rho - 3) * cos(2 * theta);
          break;
        case 12:
          wfe += weights[12] * rho * rho * (4 * rho * rho - 3) * sin(2 * theta);
          break;
        case 13:
          wfe += weights[13] * rho * (3 - 12 * rho*rho + 10 * pow(rho, 4)) *
              cos_theta;
          break;
        case 14:
          wfe += weights[14] * rho * (3 - 12 * rho*rho + 10 * pow(rho, 4)) *
              sin_theta;
          break;
        case 15:
          wfe += weights[15] * (-1 + 12 * rho*rho - 30 * pow(rho, 4) + 20 *
              pow(rho, 6));
          break;
        case 16:
          wfe += weights[16] * pow(rho, 4) * cos(4 * theta); break;
        case 17:
          wfe += weights[17] * pow(rho, 4) * sin(4 * theta); break;
        case 18:
          wfe += weights[18] * pow(rho, 3) * (5 * rho*rho - 4) * cos(3 * theta);
          break;
        case 19:
          wfe += weights[19] * pow(rho, 3) * (5 * rho*rho - 4) * sin(3 * theta);
          break;
        case 20:
          wfe += weights[20] * rho * rho * (6 - 20 * rho*rho + 15 *
              pow(rho, 4)) * cos(2 * theta);
          break;
        case 21:
          wfe += weights[21] * rho * rho * (6 - 20 * rho*rho + 15 *
              pow(rho, 4)) * sin(2 * theta);
          break;
        case 22:
          wfe += weights[22] * rho * (-4 + 30 * rho*rho - 60 * pow(rho, 4) +
              35 * pow(rho, 6)) * cos(theta);
          break;
        case 23:
          wfe += weights[23] * rho * (-4 + 30 * rho*rho - 60 * pow(rho, 4) +
              35 * pow(rho, 6)) * sin(theta);
          break;
        case 24:
          wfe += weights[24] * (1 - 20 * rho*rho + 90 * pow(rho, 4) - 140 *
              pow(rho, 6) + 70 * pow(rho, 8));
          break;
      }
    }
    output_data[i] = wfe;
  }
  
  #endif
}

#ifdef __CUDACC__
__global__
void zernike_kernel_4th(float* weights,
                        float* output,
			int size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;

  float center = 0.5 * (size - 1);

  float x = (index % size) - center;
  float y = (index / size) - center;
  float rho = sqrt(x*x + y*y) / center;
  float rho2 = rho * rho;

  float theta = atan2(y, x);
  float cos_theta = cos(theta);
  float sin_theta = sin(theta);

  float wfe =
      weights[0] +
      weights[1] * rho * cos_theta +
      weights[2] * rho * sin_theta +
      weights[3] * (rho2 - 1) +
      weights[4] * rho2 * cos(2 * theta) +
      weights[5] * rho2 * sin(2 * theta) +
      weights[6] * rho * (3 * rho2 - 2) * cos_theta +
      weights[7] * rho * (3 * rho2 - 2) * sin_theta +
      weights[8] * (1 - 6 * rho2 + 6 * rho2 * rho2);

  if (rho <= 1 && index < size * size) {
    output[index] = wfe;
  } else {
    output[index] = 0;
  }
}

#endif
