// A program to display real-time data from an SBIG detector. MTF analysis is
// also done in real time.
// Author: Philip Salvaggio

#include "mats.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <gflags/gflags.h>

#include <csignal>
#include <iostream>
#include <thread>

using namespace cv;
using namespace std;
using namespace mats;

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

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  namedWindow("Image");
  waitKey(1);
  cout << "Press 'q' to quit..." << endl;

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
  int scaled_rows = 700;
  int scaled_cols = ((double) scaled_rows * fr_width) / fr_height;

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

    resize(*frame, scaled, Size(scaled_cols, scaled_rows));
    Mat frame_uint8;
    ConvertMatToUint8(scaled, frame_uint8);
    imshow("Image", frame_uint8);

    // Push the image onto the MTF analysis queue.
    int key = waitKey(1) & 0xFF;
    if (key == 'q') keep_going = false;
  }

  image_thread.join();
  cout << "Image thread done." << endl;

  return 0;
}
