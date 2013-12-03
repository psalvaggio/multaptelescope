// File Description
// Author: Philip Salvaggio

#ifndef OPENCV_UTILS_H
#define OPENCV_UTILS_H

#include <opencv/cv.h>

cv::Mat ByteScale(const cv::Mat& input,
                  bool verbose = false);

void ByteScale(const cv::Mat& input,
               cv::Mat& output,
               bool verbose = false);

cv::Mat ByteScale(const cv::Mat& input,
                  double* min,
                  double* max,
                  bool verbose = false);

void ByteScale(const cv::Mat& input,
               cv::Mat& output,
               double* min,
               double* max,
               bool verbose = false);

cv::Mat ByteScale(const cv::Mat& input,
                  double min,
                  double max,
                  bool verbose = false);

void ByteScale(const cv::Mat& input,
               cv::Mat& output,
               double min,
               double max,
               bool verbose = false);

void LogScale(const cv::Mat& input,
              cv::Mat& output);

cv::Mat LogScale(const cv::Mat& input);

cv::Mat magnitude(const cv::Mat& input);
void magnitude(const cv::Mat& input, cv::Mat& output);


cv::Mat circshift(const cv::Mat& src,
                  cv::Point2f delta,
                  int fill=cv::BORDER_CONSTANT,
                  cv::Scalar value=cv::Scalar(0,0,0,0));

void circshift(const cv::Mat& src,
               cv::Mat& dst,
               cv::Point2f delta,
               int fill=cv::BORDER_CONSTANT,
               cv::Scalar value=cv::Scalar(0,0,0,0));

void FFTShift(const cv::Mat& input, cv::Mat& output);
cv::Mat FFTShift(const cv::Mat& input);

std::string GetMatDataType(const cv::Mat& mat);

#endif  // OPENCV_UTILS_H
