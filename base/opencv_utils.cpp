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

std::string GetMatDataType(const cv::Mat& mat) {
  int number = mat.type();

  // find type
  int imgTypeInt = number%8;
  std::string imgTypeString;

  switch (imgTypeInt) {
    case 0:
      imgTypeString = "8U";
      break;
    case 1:
      imgTypeString = "8S";
      break;
    case 2:
      imgTypeString = "16U";
      break;
    case 3:
      imgTypeString = "16S";
      break;
    case 4:
      imgTypeString = "32S";
      break;
    case 5:
      imgTypeString = "32F";
      break;
    case 6:
      imgTypeString = "64F";
      break;
    default:
      break;
  }

  // find channel
  int channel = (number/8) + 1;
  
  std::stringstream type;
  type << "CV_" << imgTypeString << "C" << channel;
 
  return type.str();
}
