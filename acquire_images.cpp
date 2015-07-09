// A program to acquire a number of images from an SBIG detector.
// Author: Philip Salvaggio

#include "mats.h"
#include <iostream>

#include <opencv2/opencv.hpp>
#include <gflags/gflags.h>

using namespace std;
using namespace cv;
using mats_io::SbigDetector;

DEFINE_bool(cooled, false, "Whether to cool the detector.");
DEFINE_int32(exposure_time, 1000, "Exposure time [hundreths of a second]");

void die(short, short) {
  cout << "Shutting down due to error." << endl;
  exit(1);
}

void AcquireImages(SbigDetector& detector,
                   int num_images,
                   const string& output_dir) {

  // Grab the size of the detector.
  uint16_t fr_width, fr_height;
  detector.GetSize(fr_width, fr_height);
  vector<uint16_t> roi{0, 0, fr_width, fr_height};


  Mat full_frame;
  for (int i = 0; i < num_images; i++) {
    detector.Capture(0, 0, fr_width, fr_height, FLAGS_exposure_time,
                     &full_frame);

    string filename =
        mats::StringPrintf("%sframe%d.png", output_dir.c_str(), i);
    imwrite(filename, full_frame);
    cout << "Acquired frame " << i << endl;
  }
}

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  // Connect to the detector.
  SbigDetector detector;
  if (detector.has_error()) return 1;
  detector.set_error_callback(die);

  if (FLAGS_cooled) {
    detector.Cool(-15);
  }

  int num_frames;
  string output_dir = ".";
  for (;;) {
    cout << "Number of images (negative to quit): ";
    cin >> num_frames;
    if (num_frames < 0) break;

    cout << "Output Directory: ";
    cin >> output_dir;

    AcquireImages(detector, num_frames, mats::AppendSlash(output_dir));
  }

  if (FLAGS_cooled) {
    detector.DisableCooling();
  }

  return 0;
}
