// File Description
// Author: Philip Salvaggio

#include "opencv_utils.h"
#include "io/logging.h"
#include <iostream>

void ByteScale(const cv::Mat& input, cv::Mat& output) {
  double min;
  double max;
  cv::minMaxIdx(input, &min, &max);
  cv::convertScaleAbs(input - min, output, 255 / (max - min));
  std::cout << "ByteScale: min = " << min << ", max = " << max << std::endl;
}

cv::Mat ByteScale(const cv::Mat& input) {
  cv::Mat output;
  ByteScale(input, output);
  return output;
}

void FFTShift(const cv::Mat& input, cv::Mat& output) {
  circshift(input, output, cv::Point2f(input.cols / 2, input.rows / 2),
            cv::BORDER_WRAP);
}

cv::Mat FFTShift(const cv::Mat& input) {
  cv::Mat output;
  FFTShift(input, output);
  return output;
}
