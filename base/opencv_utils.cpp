// File Description
// Author: Philip Salvaggio

#include "opencv_utils.h"
#include "io/logging.h"

#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <random>

using namespace cv;

Mat_<uint8_t> ByteScale(const Mat& input, bool verbose) {
  Mat_<uint8_t> output;
  ByteScale(input, output, nullptr, nullptr, verbose);
  return output;
}

void ByteScale(const Mat& input, Mat_<uint8_t>& output, bool verbose) {
  ByteScale(input, output, (double*)NULL, (double*)NULL, verbose);
}

Mat_<uint8_t> ByteScale(const Mat& input,
                        double* min,
                        double* max,
                        bool verbose) {
  Mat_<uint8_t> output;
  ByteScale(input, output, min, max, verbose);
  return output;
}

void ByteScale(const Mat& input,
               Mat_<uint8_t>& output,
               double* min,
               double* max,
               bool verbose) {
  double local_min;
  double local_max;
  minMaxIdx(input, &local_min, &local_max);

  if (min != NULL) *min = local_min;
  if (max != NULL) *max = local_max;

  ByteScale(input, output, local_min, local_max, verbose);
}

Mat_<uint8_t> ByteScale(const Mat& input,
                        double min,
                        double max,
                        bool verbose) {
  Mat_<uint8_t> output;
  ByteScale(input, output, min, max, verbose);
  return output;
}

void ByteScale(const Mat& input,
               Mat_<uint8_t>& output,
               double min,
               double max,
               bool verbose) {
  convertScaleAbs(input - min, output, 255 / (max - min));
  if (verbose) {
    std::cout << "ByteScale: min = " << min << ", max = " << max << std::endl;
  }
}

void LogScale(const Mat& input, Mat_<uint8_t>& output) {
  Mat log_input;
  log(input + 1, log_input);
  ByteScale(log_input, output);
}

Mat_<uint8_t> LogScale(const Mat& input) {
  Mat_<uint8_t> output;
  LogScale(input, output);
  return output;
}

Mat GammaScale(const Mat& input, double gamma) {
  double min;
  double max;
  minMaxIdx(input, &min, &max);

  Mat scaled;
  input.convertTo(scaled, CV_64F);
  scaled = (scaled - min) / (max - min);
  pow(scaled, gamma, scaled);
  scaled *= 255;
  scaled.convertTo(scaled, CV_8U);
  return scaled;
}

Mat_<Vec3b> ColorScale(const Mat& input, int colormap) {
  Mat_<Vec3b> output;
  applyColorMap(ByteScale(input), output, colormap);
  return output;
}

Mat_<double> magnitude(const Mat_<std::complex<double>>& input) {
  Mat_<double> output;
  magnitude(input, output);
  return output;
}

void magnitude(const Mat_<std::complex<double>>& input, Mat_<double>& output) {
  std::vector<Mat> input_planes;
  split(input, input_planes);
  magnitude(input_planes.at(0), input_planes.at(1), output);
}

void FFTShift(const Mat& input, Mat& output) {
  circshift(input, output, Point2f(input.cols / 2,
                                   input.rows / 2), BORDER_WRAP);
}

Mat FFTShift(const Mat& input) {
  Mat output;
  FFTShift(input, output);
  return output;
}

void IFFTShift(const Mat& input, Mat& output) {
  circshift(input, output, Point2f(-input.cols / 2,
                                   -input.rows / 2), BORDER_WRAP);
}

Mat IFFTShift(const Mat& input) {
  Mat output;
  IFFTShift(input, output);
  return output;
}

std::string GetMatDataType(const Mat& mat) {
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

void ConvertMatToDouble(const Mat& input, Mat& output) {
  double scale_factor = 1;

  int image_type_int = input.type() % 8;

  switch (image_type_int) {
    case 0: case 1: // 8U, 8S
      scale_factor = 1. / 255;
      break;
    case 2: case 3: // 16S, 16U  
      scale_factor = 1. / 65535;
      break;
    case 4: // 32S
      scale_factor = 1. / (pow(2., 32) - 1);
      break;
  }

  input.convertTo(output, CV_64F, scale_factor);
}

void ConvertMatToUint8(const Mat& input, Mat& output) {
  double scale_factor = 1;

  int image_type_int = input.type() % 8;

  switch (image_type_int) {
    case 0: case 1: // 8U, 8S
      scale_factor = 1;
      break;
    case 2: case 3: // 16S, 16U  
      scale_factor = 1. / 256;
      break;
    case 4: // 32S
      scale_factor = 1. / pow(2., 24);
      break;
    case 5: case 6: // 32F, 64F
      scale_factor = 255;
      break;
  }

  input.convertTo(output, CV_8U, scale_factor);
}

void GetRadialProfile(const Mat& input, double theta,
                      std::vector<double>* output) {
  if (!output) return;
  output->clear();

  const int rows = input.rows;
  const int cols = input.cols;

  int profile_size = std::min(rows, cols) / 2;
  int center_x = (cols + 1) / 2;
  int center_y = (rows + 1) / 2;

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

struct RoiRectangleData {
  std::vector<uint16_t>* corners;
  Mat* image;
  std::string* window;
};

// Grab the user clicks on the window for ROI acquisition.
void mouse_callback(int event, int x, int y, int, void* userdata) {
  auto roi_data = reinterpret_cast<RoiRectangleData*>(userdata);

  auto draw_roi =
    [roi_data] (int x0, int y0, int x1, int y1) {
      Mat annotated_image;
      cvtColor(*(roi_data->image), annotated_image, CV_GRAY2BGR);
      rectangle(annotated_image, Point(x0, y0), Point(x1, y1),
                Scalar(0, 255, 0), 1);
      imshow(*(roi_data->window), annotated_image);
    };

  if (event == EVENT_LBUTTONDOWN) {
    roi_data->corners->clear();
    roi_data->corners->push_back(static_cast<uint16_t>(x));
    roi_data->corners->push_back(static_cast<uint16_t>(y));
  } else if (event == EVENT_MOUSEMOVE) {
    if (roi_data->corners->size() == 2) {
      draw_roi(roi_data->corners->at(0), roi_data->corners->at(1), x, y);
    }
  } else {
    roi_data->corners->push_back(static_cast<uint16_t>(x));
    roi_data->corners->push_back(static_cast<uint16_t>(y));
    draw_roi(roi_data->corners->at(0), roi_data->corners->at(1), x, y);
  }
}

std::vector<uint16_t> GetRoi(const Mat& image) {
  int rows = image.rows, cols = image.cols;
  double aspect_ratio = static_cast<double>(rows) / cols;

  // Compute the scaled size of the image so that it can fit on the screen.
  int scale_rows = std::min(770, rows);
  int scale_cols = scale_rows / aspect_ratio;
  double scale_factor = static_cast<double>(rows) / scale_rows;

  // Resample the image to display.
  Mat display_image;
  resize(image, display_image, Size(scale_cols, scale_rows));
  display_image = ByteScale(display_image);

  // Show the full-frame image.
  std::string input_window = "Drag to specify an ROI. Enter to confirm.";
  namedWindow(input_window, CV_GUI_NORMAL | WINDOW_AUTOSIZE);
  moveWindow(input_window, 0, 0);
  imshow(input_window, display_image);

  // Wait for the user to select the corners.
  std::vector<uint16_t> roi;
  RoiRectangleData roi_data{&roi, &display_image, &input_window};
  setMouseCallback(input_window, mouse_callback, &roi_data);
  int key = 0;
  while (key != 13 || roi.size() != 4) {
    key = waitKey(50) & 0xff;
  }
  destroyWindow(input_window);
  waitKey(1);

  // Scale back to the image's resolution.
  for (auto& tmp : roi) tmp *= scale_factor;
  roi[2] = roi[2] - roi[0] + 1;
  roi[3] = roi[3] - roi[1] + 1;

  return roi;
}

Mat_<double> CreateEdgeTarget(int width,
                                  int height,
                                  double angle,
                                  double dim,
                                  double bright,
                                  double noise) {
  double nx = cos(angle * M_PI / 180);
  double ny = sin(angle * M_PI / 180);

  std::default_random_engine rng;
  std::normal_distribution<double> noise_dist(0, noise);

  double offset = (width / 2.0) * nx + (height / 2.0) * ny;

  Mat_<double> image(height, width, CV_64FC1);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      double nDotR = j * nx + i * ny;
      double edge_offset = nDotR - offset;

      double value = 0;
      if (edge_offset > 0.5) {
        value = bright;
      } else if (edge_offset < -0.5) {
        value = dim;
      } else {
        value = (edge_offset + 0.5) * (bright - dim) + dim;
      }

      image(i, j) = value + noise_dist(rng);
    }
  }

  return image;
}

void rotate(const Mat& src, double angle, Mat& dst) {
  int len = std::max(src.cols, src.rows);
  Point2f pt(0.5 * len, 0.5 * len);
  Mat r = getRotationMatrix2D(pt, angle, 1.0);

  warpAffine(src, dst, r, Size(len, len));
}
