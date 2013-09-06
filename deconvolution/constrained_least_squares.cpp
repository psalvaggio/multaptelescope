// File Description
// Author: Philip Salvaggio

#include "constrained_least_squares.h"
#include "base/opencv_utils.h"

#include <opencv/highgui.h>

using namespace cv;

ConstrainedLeastSquares::ConstrainedLeastSquares() {}

void ConstrainedLeastSquares::Deconvolve(const Mat& input,
                                         const Mat& transfer_function,
                                         double smoothness,
                                         Mat* output) {
  Mat input_fft;
  dft(input, input_fft, DFT_COMPLEX_OUTPUT);

  vector<Mat> input_fft_planes;
  split(input_fft, input_fft_planes);

  Mat laplacian = Mat::zeros(input.rows, input.cols, CV_64FC1);
  double* laplacian_data = (double*) laplacian.data;
  laplacian_data[0*input.cols + 0] = 4;
  laplacian_data[1*input.cols + 0] = -1;
  laplacian_data[0*input.cols + 1] = -1;
  laplacian_data[0*input.rows + input.cols - 1] = -1;
  laplacian_data[(input.rows - 1)*input.cols + 0] = -1;

  Mat laplacian_fft;
  dft(laplacian, laplacian_fft, DFT_COMPLEX_OUTPUT);

  vector<Mat> laplacian_fft_planes;
  split(laplacian_fft, laplacian_fft_planes);

  vector<Mat> transfer_conj_planes;
  split(transfer_function, transfer_conj_planes);

  Mat transfer_power = transfer_conj_planes[0].mul(transfer_conj_planes[0]) +
                       transfer_conj_planes[1].mul(transfer_conj_planes[1]);
  Mat smooth_power = laplacian_fft_planes[0].mul(laplacian_fft_planes[0]) +
                     laplacian_fft_planes[1].mul(laplacian_fft_planes[1]);
  Mat denominator = transfer_power + smoothness * smooth_power;
  
  divide(transfer_conj_planes[0], denominator, transfer_conj_planes[0]);
  divide(transfer_conj_planes[1], denominator, transfer_conj_planes[1], -1);

  vector<Mat> output_planes;
  output_planes.push_back(transfer_conj_planes[0].mul(input_fft_planes[0]) -
                          transfer_conj_planes[1].mul(input_fft_planes[1]));
  output_planes.push_back(transfer_conj_planes[0].mul(input_fft_planes[1]) +
                          transfer_conj_planes[1].mul(input_fft_planes[0]));

  Mat output_fft;
  merge(output_planes, output_fft);
  dft(output_fft, *output, DFT_REAL_OUTPUT | DFT_INVERSE | DFT_SCALE);
}
