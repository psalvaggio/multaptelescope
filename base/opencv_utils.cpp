// File Description
// Author: Philip Salvaggio

#include "opencv_utils.h"
#include "io/logging.h"
#include <iostream>

cv::Mat ByteScale(const cv::Mat& input,
                  bool verbose) {
  cv::Mat output;
  ByteScale(input, output, (double*)NULL, (double*)NULL, verbose);
  return output;
}

void ByteScale(const cv::Mat& input,
               cv::Mat& output,
               bool verbose) {
  ByteScale(input, output, (double*)NULL, (double*)NULL, verbose);
}

cv::Mat ByteScale(const cv::Mat& input,
                  double* min,
                  double* max,
                  bool verbose) {
  cv::Mat output;
  ByteScale(input, output, min, max, verbose);
  return output;
}

void ByteScale(const cv::Mat& input,
               cv::Mat& output,
               double* min,
               double* max,
               bool verbose) {
  double local_min;
  double local_max;
  cv::minMaxIdx(input, &local_min, &local_max);

  if (min != NULL) *min = local_min;
  if (max != NULL) *max = local_max;

  ByteScale(input, output, local_min, local_max, verbose);
}

cv::Mat ByteScale(const cv::Mat& input,
                  double min,
                  double max,
                  bool verbose) {
  cv::Mat output;
  ByteScale(input, output, min, max, verbose);
  return output;
}

void ByteScale(const cv::Mat& input,
               cv::Mat& output,
               double min,
               double max,
               bool verbose) {
  cv::convertScaleAbs(input - min, output, 255 / (max - min));
  if (verbose) {
    std::cout << "ByteScale: min = " << min << ", max = " << max << std::endl;
  }
}

void LogScale(const cv::Mat& input,
              cv::Mat& output) {
  log(input + 1, output);
  ByteScale(output, output);
}

cv::Mat LogScale(const cv::Mat& input) {
  cv::Mat output;
  LogScale(input, output);
  return output;
}

cv::Mat GammaScale(const cv::Mat& input, double gamma) {
  double min;
  double max;
  cv::minMaxIdx(input, &min, &max);

  cv::Mat scaled;
  input.convertTo(scaled, CV_64F);
  scaled = (scaled - min) / (max - min);
  cv::pow(scaled, gamma, scaled);
  scaled *= 255;
  scaled.convertTo(scaled, CV_8U);
  return scaled;
}

cv::Mat magnitude(const cv::Mat& input) {
  cv::Mat output;
  magnitude(input, output);
  return output;
}

void magnitude(const cv::Mat& input, cv::Mat& output) {
  std::vector<cv::Mat> input_planes;
  cv::split(input, input_planes);
  cv::magnitude(input_planes.at(0), input_planes.at(1), output);
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
