// File Description
// Author: Philip Salvaggio

#include "photon_noise.h"

using namespace cv;
using std::vector;

namespace mats {

PhotonNoise::PhotonNoise() {}
PhotonNoise::~PhotonNoise() {}

void PhotonNoise::AddPhotonNoise(cv::Mat* signal) {
  if (!signal) return;

  AddPhotonNoise(*signal, signal);
}

void PhotonNoise::AddPhotonNoise(const cv::Mat& input, cv::Mat* output) {
  if (!output) return;

  if (input.rows != output->rows || input.cols != output->cols ||
      input.type() != output->type()) {
    output->create(input.rows, input.cols, input.type());
  }

  const double* in_data = (const double*)input.data;
  double* out_data = (double*)output->data;
  size_t size = input.rows * input.cols;

  Mat random(input.rows, input.cols, CV_64FC1);
  randn(random, 0, 1);
  double* random_data = (double*)random.data;

  for (size_t i = 0; i < size; i++) {
    out_data[i] = in_data[i] + sqrt(in_data[i]) * random_data[i];
  }
}

}
