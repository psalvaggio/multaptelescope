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

  Mat_<double> random(input.size());
  randn(random, 0, 1);

  Mat& noisy(*output);

  for (int i = 0; i < input.rows; i++) {
    for (int j = 0; j < input.cols; j++) {
      double input_val = std::max(input.at<double>(i, j), 0.);
      noisy.at<double>(i, j) = input_val + sqrt(input_val) * random(i, j);
    }
  }
}

}
