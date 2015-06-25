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

void GetRadialProfile(const cv::Mat& input, double theta,
                      std::vector<double>* output) {
  if (!output) return;
  output->clear();

  const int rows = input.rows;
  const int cols = input.cols;

  int profile_size = std::min(rows, cols) / 2;
  int center_x = cols / 2;
  int center_y = rows / 2;

  double dx = cos(theta);
  double dy = sin(theta);

  output->reserve(profile_size);
  for (int i = 0; i < profile_size; i++) {
    double x = center_x + i * dx;
    double y = center_y + i * dy;

    int x_lt = static_cast<int>(x);
    int x_gt = x_lt + 1;
    int y_lt = static_cast<int>(y);
    int y_gt = y_lt + 1;

    if (x_lt > 0 && y_lt > 0 && x_gt < cols && y_gt < rows) {
      double alpha_x = x - x_lt;
      double alpha_y = y - y_lt;
      double inter_y_lt = (1-alpha_y) * input.at<double>(y_lt, x_lt) +
                          alpha_y * input.at<double>(y_gt, x_lt);
      double inter_y_gt = (1-alpha_y) * input.at<double>(y_lt, x_gt) +
                          alpha_y * input.at<double>(y_gt, x_gt);
      output->push_back((1-alpha_x) * inter_y_lt + alpha_x * inter_y_gt);
    } else {
      int x_rnd = std::max(std::min((int)round(x), cols - 1), 0);
      int y_rnd = std::max(std::min((int)round(y), rows - 1), 0);
      output->push_back(input.at<double>(y_rnd, x_rnd));
    }
  }
}

// Grab the user clicks on the window for ROI acquisition.
void mouse_callback(int event, int x, int y, int, void* userdata) {
  auto corners = reinterpret_cast<std::vector<uint16_t>*>(userdata);
  if (event == cv::EVENT_LBUTTONDOWN) {
    corners->push_back(static_cast<uint16_t>(x));
    corners->push_back(static_cast<uint16_t>(y));
  }
}

std::vector<uint16_t> GetRoi(const cv::Mat& image) {
  int rows = image.rows, cols = image.cols;
  double aspect_ratio = static_cast<double>(rows) / cols;

  // Compute the scaled size of the image so that it can fit on the screen.
  int scale_cols = 1080;
  int scale_rows = aspect_ratio * scale_cols;
  double scale_factor = static_cast<double>(rows) / scale_rows;

  // Resample the image to display.
  cv::Mat display_image;
  cv::resize(image, display_image, cv::Size(scale_cols, scale_rows));

  // Show the full-frame image.
  std::string input_window = "Please click on the corners of the ROI";
  cv::namedWindow(input_window, CV_GUI_NORMAL | cv::WINDOW_AUTOSIZE);
  cv::moveWindow(input_window, 0, 0);
  cv::imshow(input_window, ByteScale(display_image));

  // Wait for the user to select the corners.
  std::vector<uint16_t> roi;
  cv::setMouseCallback(input_window, mouse_callback, &roi);
  while (roi.size() != 4) {
    cv::waitKey(50);
  }

  // Scale back to the image's resolution.
  std::for_each(std::begin(roi), std::end(roi),
      [scale_factor] (uint16_t& i) { i *= scale_factor; });
  roi[2] = roi[2] - roi[0] + 1;
  roi[3] = roi[3] - roi[1] + 1;

  cv::destroyWindow(input_window);

  return roi;
}
