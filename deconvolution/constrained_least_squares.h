// A simple deconvolution filter based on the Wiener filter. The filter is of
// the form
//                              H*(xi, eta)
// W(xi, eta) = --------------------------------------------
//              |H(xi, eta)|^2 + smoothness * |S(xi, eta)|^2
//
// where H is the OTF of the system, smoothness is a parameter that smooths the
// reconstruction, and S is a smoothness spectrum (FFT of a Laplacian
// sharpening kernel). Typical values of smoothness are 1e-3 for an aggressive
// reconstruction or 1e-2 for more noise robustness.
// Author: Philip Salvaggio

#ifndef CONSTRAINED_LEAST_SQUARES_H
#define CONSTRAINED_LEAST_SQUARES_H

#include <opencv2/core/core.hpp>

class ConstrainedLeastSquares {
 public:
  ConstrainedLeastSquares();

  using ComplexMat = cv::Mat_<std::complex<double>>;

  void Deconvolve(const cv::Mat_<double>& input,
                  const ComplexMat& transfer_function,
                  double smoothness,
                  cv::Mat_<double>* output);

  void GetInverseFilter(const ComplexMat& transfer_function,
                        double smoothness,
                        ComplexMat* output);
};

#endif  // CONSTRAINED_LEAST_SQUARES_H
