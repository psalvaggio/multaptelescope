// File Description
// Author: Philip Salvaggio

#ifndef CONSTRAINED_LEAST_SQUARES_H
#define CONSTRAINED_LEAST_SQUARES_H

#include <opencv2/core/core.hpp>

class ConstrainedLeastSquares {
 public:
  ConstrainedLeastSquares();

  void Deconvolve(const cv::Mat& input,
                  const cv::Mat& transfer_function,
                  double smoothness,
                  cv::Mat* output);

  void GetInverseFilter(const cv::Mat& transfer_function,
                        double smoothness,
                        cv::Mat* output);
};

#endif  // CONSTRAINED_LEAST_SQUARES_H
