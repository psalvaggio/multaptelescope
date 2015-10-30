// Since the sparse aperture mask results in vignetting, we can use the
// vignetting pattern to center the mask. It is assumed that the system is
// looking at a relatively constant region.
// Author: Philip Salvaggio

#include "mats.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <csignal>
#include <iostream>
#include <thread>
#include <unistd.h>

#include <gflags/gflags.h>

using namespace cv;
using namespace std;

DEFINE_int32(exposure_time, 9, "Exposure Time [hundreths of a second]");

static bool keep_going = true;
static std::thread image_thread;

void die(short, short) {
  cout << "Shutting down due to error." << endl;
  exit(1);
}


void CaptureSigint(int) {
  keep_going = false;
  image_thread.join();
  exit(0);
}


// Find the center of the vignetting pattern. This is done by taking the sums of
// each row and column, subtracting off the background noise level, treating the
// row/column sums as a PDF and finding the expected value.
bool FindVignettingCenter(const Mat& image,
                          double& center_x,
                          double& center_y) {
  vector<double> row_hist(image.rows, 0), col_hist(image.cols, 0);

  double row_min = 1e10, col_min = 1e10;
  double row_total = 0, col_total = 0;
  for (int i = 0; i < image.rows; i++) {
    row_hist[i] = (sum(image.row(i)))[0];
    row_min = min(row_hist[i], row_min);
    row_total += row_hist[i];
  }
  for (int i = 0; i < image.cols; i++) {
    col_hist[i] = (sum(image.col(i)))[0];
    col_min = min(col_hist[i], col_min);
    col_total += col_hist[i];
  }

  row_total -= row_min * row_hist.size();
  col_total -= col_min * col_hist.size();

  center_y = 0;
  for (int i = 0; i < image.rows; i++) {
    center_y += i * (row_hist[i] - row_min) / row_total;
  }

  center_x = 0;
  for (int i = 0; i < image.cols; i++) {
    center_x += i * (col_hist[i] - col_min) / col_total;
  }
  return true;
}


int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  CHECK(FLAGS_exposure_time >= 9,
        "Error: Exposure time must be at least 0.09 seconds.");

  namedWindow("Image");
  waitKey(1);

  WaitQueue<Mat> image_queue(1);

  signal(SIGINT, CaptureSigint);

  // Connect to the detector.
  mats_io::SbigDetector detector;
  if (detector.has_error()) {
    keep_going = false;
    return 1;
  }
  detector.set_error_callback(die);

  // Grab the size of the detector.
  uint16_t fr_width, fr_height;
  detector.GetSize(fr_width, fr_height);
  vector<uint16_t> roi{0, 0, fr_width, fr_height};

  Mat scaled;
  int scaled_rows = 770;
  int scaled_cols = ((double) scaled_rows * fr_width) / fr_height;

  // Dispatch the image acquisition thread.
  image_thread = thread(mats_io::SbigAcquisitionThread,
      &image_queue, &detector, &roi, &FLAGS_exposure_time, &keep_going);

  while (keep_going) {
    // Grab an image from the current ROI and show a preview.
    Mat* raw_image_ptr = image_queue.wait_for(100);
    if (raw_image_ptr == nullptr) continue;

    unique_ptr<Mat> frame(raw_image_ptr);
    if (frame->rows != roi[3] || frame->cols != roi[2]) {
      continue;
    }

    // Overlay the crosshairs for the image center.
    resize(*frame, scaled, Size(scaled_cols, scaled_rows));
    cvtColor(ByteScale(scaled), scaled, CV_GRAY2BGR, 3);
    scaled.row(scaled_rows / 2).setTo(Scalar(0, 255, 0));
    scaled.col(scaled_cols / 2).setTo(Scalar(0, 255, 0));

    // Find the center and draw the displacement vector to the center.
    double center_x, center_y;
    if (FindVignettingCenter(*frame, center_x, center_y)) {
      double scale = scaled_rows / double(frame->rows);
      arrowedLine(scaled, Point(center_x * scale, center_y * scale),
          Point(scaled_cols / 2, scaled_rows / 2), Scalar(0, 0, 255), 1); 
    }
    imshow("Image", scaled);
    waitKey(1);
  }

  image_thread.join();
  cout << "Image thread done." << endl;

  return 0;
}
