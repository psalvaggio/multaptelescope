// Implementation file for constrained_least_squares.h.
// Author: Philip Salvaggio

#include "constrained_least_squares.h"
#include "base/opencv_utils.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>

using namespace cv;
using std::vector;

ConstrainedLeastSquares::ConstrainedLeastSquares() {}

void ConstrainedLeastSquares::Deconvolve(const Mat_<double>& input,
                                         const ComplexMat& transfer_function,
                                         double smoothness,
                                         Mat_<double>* output) {
  ComplexMat input_fft;
  dft(input, input_fft, DFT_COMPLEX_OUTPUT);

  ComplexMat otf(transfer_function);
  if (input.rows != otf.rows || input.cols != otf.cols) {
    resize(transfer_function, otf, input.size());
  }

  ComplexMat inv_filter, output_fft;
  GetInverseFilter(otf, smoothness, &inv_filter);
  mulSpectrums(input_fft, inv_filter, output_fft, 0, false);

  dft(output_fft, *output, DFT_INVERSE | DFT_SCALE | DFT_REAL_OUTPUT);
}

void ConstrainedLeastSquares::GetInverseFilter(
    const ComplexMat& transfer_function,
    double smoothness,
    ComplexMat* output) {
  Mat_<double> laplacian = Mat::zeros(transfer_function.rows,
                                      transfer_function.cols, CV_64FC1);
  laplacian(0, 0) = 4;
  laplacian(0, 1) = -1;
  laplacian(1, 0) = -1;
  laplacian(0, transfer_function.cols - 1) = -1;
  laplacian(transfer_function.rows - 1, 0) = -1;

  Mat laplacian_fft;
  dft(laplacian, laplacian_fft, DFT_COMPLEX_OUTPUT);

  vector<Mat_<double>> laplacian_fft_planes;
  split(laplacian_fft, laplacian_fft_planes);
  Mat_<double> smooth_power =
      laplacian_fft_planes[0].mul(laplacian_fft_planes[0]) +
      laplacian_fft_planes[1].mul(laplacian_fft_planes[1]);

  vector<Mat_<double>> transfer_conj_planes;
  split(transfer_function, transfer_conj_planes);
  Mat_<double> transfer_power =
      transfer_conj_planes[0].mul(transfer_conj_planes[0]) +
      transfer_conj_planes[1].mul(transfer_conj_planes[1]);

  Mat_<double> denominator = transfer_power + smooth_power * smoothness;

  vector<Mat_<double>> inv_filter;
  split(transfer_function, inv_filter);
  inv_filter[1] *= -1;
  divide(inv_filter[0], denominator, inv_filter[0]);
  divide(inv_filter[1], denominator, inv_filter[1]);
  merge(inv_filter, *output);
}
