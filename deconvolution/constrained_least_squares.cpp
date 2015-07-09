// File Description
// Author: Philip Salvaggio

#include "constrained_least_squares.h"
#include "base/opencv_utils.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>

using namespace cv;
using std::vector;

ConstrainedLeastSquares::ConstrainedLeastSquares() {}

void ConstrainedLeastSquares::Deconvolve(const Mat& input,
                                         const Mat& transfer_function,
                                         double smoothness,
                                         Mat* output) {
  Mat input_fft;
  dft(input, input_fft, DFT_COMPLEX_OUTPUT);

  Mat inv_filter;
  GetInverseFilter(transfer_function, smoothness, &inv_filter);
  mulSpectrums(input_fft, inv_filter, *output, 0, true);
}

void ConstrainedLeastSquares::GetInverseFilter(const Mat& transfer_function,
                                               double smoothness,
                                               Mat* output) {
  Mat_<double> laplacian = Mat::zeros(transfer_function.rows,
                                      transfer_function.cols, CV_64FC1);
  laplacian(0, 0) = 4;
  laplacian(0, 1) = -1;
  laplacian(1, 0) = -1;
  laplacian(0, transfer_function.cols - 1) = -1;
  laplacian(transfer_function.rows - 1, 0) = -1;

  Mat laplacian_fft;
  dft(laplacian, laplacian_fft, DFT_COMPLEX_OUTPUT);

  vector<Mat> laplacian_fft_planes;
  split(laplacian_fft, laplacian_fft_planes);
  Mat smooth_power = laplacian_fft_planes[0].mul(laplacian_fft_planes[0]) +
                     laplacian_fft_planes[1].mul(laplacian_fft_planes[1]);

  vector<Mat> transfer_conj_planes;
  split(transfer_function, transfer_conj_planes);
  Mat transfer_power = transfer_conj_planes[0].mul(transfer_conj_planes[0]) +
                       transfer_conj_planes[1].mul(transfer_conj_planes[1]);

  Mat denominator = transfer_power + smooth_power * smoothness;

  vector<Mat> inv_filter;
  split(transfer_function, inv_filter);
  inv_filter[1] *= -1;
  divide(inv_filter[0], denominator, inv_filter[0]);
  divide(inv_filter[1], denominator, inv_filter[1]);
  merge(inv_filter, *output);
}
