// A program to acquire a number of images from an SBIG detector.
// Author: Philip Salvaggio

#include "mats.h"
#include <iostream>

#include <opencv2/opencv.hpp>
#include <gflags/gflags.h>

using namespace std;
using namespace cv;
using mats_io::SbigDetector;

DEFINE_int32(exposure_time, 1000, "Exposure time [hundreths of a second]");

static int frame_index = 0;

void die(short, short) {
  cout << "Shutting down due to error." << endl;
  exit(1);
}

void AcquireImages(SbigDetector& detector) {
  cout << "Acquisition Type:" << endl
       << " n > 0: Acquire n images" << endl
       << " n = 0: On-demand (Enter to capture, q to quit)" << endl
       << " n < 0: Abort" << endl;

  int num_images;
  string output_dir = ".";

  cout << "Acquisition Type: ";
  cin >> num_images;
  if (num_images < 0) return;

  cout << "Output Directory: ";
  cin >> output_dir;

  output_dir = mats::ResolvePath(output_dir);

  // Grab the size of the detector.
  uint16_t fr_width, fr_height;
  detector.GetSize(fr_width, fr_height);
  vector<uint16_t> roi{0, 0, fr_width, fr_height};

  auto Save = [&output_dir] (Mat& image, int frame_num) {
    string filename =
        mats::StringPrintf("%sframe%03d.png", output_dir.c_str(), frame_num);
    imwrite(filename, image);
    cout << "Acquired frame " << frame_num << endl;
  };

  Mat full_frame;
  if (num_images == 0) {
    cin.get();
    int key;
    do {
      key = cin.get();
      if (key == '\n') {
        detector.Capture(0, 0, fr_width, fr_height, FLAGS_exposure_time,
                         &full_frame);

        Save(full_frame, frame_index++);
      }
    } while (key != 'q');
  } else {
    for (int i = 0; i < num_images; i++) {
      detector.Capture(0, 0, fr_width, fr_height, FLAGS_exposure_time,
                       &full_frame);

      Save(full_frame, frame_index++);
    }
  }
}

void OutputMenu() {
  cout << "Commands:" << endl
       << "  acquire: Acquire image" << endl
       << "  set_exposure: Set the exposure time" << endl
       << "  cool: Cool the detector" << endl
       << "  warm: Warm the detector" << endl
       << "  help: Reprint this menu" << endl
       << "  exit: Quit" << endl;
}

void SetIntegrationTime() {
  cout << "Integration Time [hundreths of a second]: ";

  int int_time;
  if (cin >> int_time) {
    if (int_time >= 9) {
      FLAGS_exposure_time = int_time;
    } else {
      cerr << "Error: integration time must be greater than .09 seconds."
           << endl;
    }
  } else {
    cerr << "Invalid Value." << endl;
  }
}

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  // Connect to the detector.
  SbigDetector detector;
  if (detector.has_error()) return 1;
  detector.set_error_callback(die);

  bool is_cooled = false;

  OutputMenu();
  string command;
  do {
    cout << "> ";
    cin >> command;
    if (command == "acquire") {
      AcquireImages(detector);
    } else if (command == "set_exposure") {
      SetIntegrationTime();
    } else if (command == "cool") {
      if (!is_cooled) detector.Cool(-15);
      is_cooled = true;
    } else if (command == "warm") {
      if (is_cooled) detector.DisableCooling();
      is_cooled = false;
    } else if (command == "help") {
      OutputMenu();
    } else if (command == "exit") {
      break;
    } else {
      cerr << "Unrecognized command: " << command << endl;
    }
  } while (true);

  if (is_cooled) detector.DisableCooling();

  return 0;
}
