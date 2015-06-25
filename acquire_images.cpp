// A program to acquire a number of images from an SBIG detector.
// Author: Philip Salvaggio

#include "mats.h"
#include <iostream>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void die(short, short) {
  cout << "Shutting down due to error." << endl;
  exit(1);
}

void AcquireImages(int num_images, const string& output_dir) {
  // Connect to the detector.
  mats_io::SbigDetector detector;
  if (detector.has_error()) return;
  detector.set_error_callback(die);

  // Grab the size of the detector.
  uint16_t fr_width, fr_height;
  detector.GetSize(fr_width, fr_height);
  vector<uint16_t> roi{0, 0, fr_width, fr_height};

  const int kExposureTime = 9; // [hundreths of a second]

  Mat full_frame;
  for (int i = 0; i < num_images; i++) {
    detector.Capture(0, 0, fr_width, fr_height, kExposureTime, &full_frame);

    string filename =
        mats::StringPrintf("%sframe%d.png", output_dir.c_str(), i);
    imwrite(filename, full_frame);
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    cerr << "Usage: " << argv[0] << " num_images output_dir" << endl;
    return 1;
  }

  AcquireImages(atoi(argv[1]), mats::AppendSlash(argv[2]));

  return 0;
}
