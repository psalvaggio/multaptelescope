// File Description
// Author: Philip Salvaggio

#ifndef OPENCV_UTILS_H
#define OPENCV_UTILS_H

#include <opencv/cv.h>

void ByteScale(const cv::Mat& input, cv::Mat& output);
cv::Mat ByteScale(const cv::Mat& input);


void circshift(const cv::Mat& src,
               cv::Mat& dst,
               cv::Point2f delta,
               int fill=cv::BORDER_CONSTANT,
               cv::Scalar value=cv::Scalar(0,0,0,0));

void FFTShift(const cv::Mat& input, cv::Mat& output);
cv::Mat FFTShift(const cv::Mat& input);


#endif  // OPENCV_UTILS_H
