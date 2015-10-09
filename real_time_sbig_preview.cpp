// A program to display real-time data from an SBIG detector. MTF analysis is
// also done in real time.
// Author: Philip Salvaggio

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "mats.h"

#include <csignal>
#include <iostream>
#include <thread>
#include <queue>
#include <unistd.h>

#include <gflags/gflags.h>

using namespace cv;
using namespace std;
using mats::MenuApplication;

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

// Run the menu application.
void InputThread(MenuApplication* app) {
  app->run();
}

void RunMTFAnalysis(WaitQueue<Mat>* image_queue,
                    const mats::Telescope* telescope,
                    bool* average_mode,
                    bool* keep_going,
                    bool* save_result) {
  if (!image_queue || !keep_going) return;

  Gnuplot gp;

  // Set up the spectral resolution of the simulation.
  vector<vector<double>> raw_weighting;
  mats_io::TextFileReader::Parse(
      telescope->sim_params().spectral_weighting_filename(),
      &raw_weighting);
  const vector<double>& wavelengths(raw_weighting[0]);
  const vector<double>& spectral_weighting(raw_weighting[1]);

  // Compute the theoretical 2D OTF of the telescope.
  Mat theoretical_otf;
  telescope->ComputeEffectiveOtf(wavelengths,
                                 spectral_weighting,
                                 &theoretical_otf);
  Mat theoretical_2d_mtf = magnitude(theoretical_otf);

  // Grab the radial profile of the OTF and convert to MTF.
  std::vector<double> theoretical_mtf;
  GetRadialProfile(FFTShift(theoretical_2d_mtf), 0, &theoretical_mtf);
  
  vector<pair<double, double>> theoretical_mtf_data;
  for (size_t i = 0; i < theoretical_mtf.size(); i++) {
    theoretical_mtf_data.emplace_back(i / (2. * (theoretical_mtf.size() - 1)),
                                      theoretical_mtf[i]);
  }

  double edge_angle = 0;
  std::vector<double> mtf;
  int mtf_count = 0;
  SlantEdgeMtf slant_edge;

  while (*keep_going) {
    usleep(1e5);
    // Wait until we get a new image.
    Mat* raw_image_ptr = image_queue->wait_for(100);
    if (raw_image_ptr == nullptr) continue;
    unique_ptr<Mat> image(raw_image_ptr);

    // Perform MTF analysis on the new image.
    if (*average_mode) {
      vector<double> tmp_mtf;
      slant_edge.Analyze(*image, &edge_angle, &tmp_mtf);
      for (size_t i = 0; i < tmp_mtf.size(); i++) {
        if (i < mtf.size()) {
          mtf[i] = (mtf_count * mtf[i] + tmp_mtf[i]) / (mtf_count + 1);
        } else {
          mtf.push_back(tmp_mtf[i]);
        }
      }
    } else {
      mtf.clear();
      slant_edge.Analyze(*image, &edge_angle, &mtf);
      mtf_count = 1;
    }

    // Set up the data for plotting the 2 MTFs.
    vector<pair<double, double>> measured_mtf_data;//, theoretical_mtf_data;
    for (size_t j = 0; j < mtf.size(); j++) {
      measured_mtf_data.emplace_back(j / (2. * (mtf.size() - 1)), mtf[j]);
    }

    // Plot the MTfs.
    gp << "set xlabel \"Spatial Frequency [cyc / pixel]\"\n"
       << "set ylabel \"MTF\"\n"
       << "set xrange [0:0.5]\n"
       << "set yrange [0:1]\n"
       << "plot " << gp.file1d(measured_mtf_data) << "w l t \"Measured\" lw 3, "
       << gp.file1d(theoretical_mtf_data) << "w l t \"Predicted\" lw 3\n";
    gp << endl;

    // Optionally save the result.
    if (*save_result) {
      imwrite("edge.png", *image);
      ofstream ofs("mtf.txt");
      if (ofs) {
        for (size_t j = 0; j < measured_mtf_data.size(); j++) {
          ofs << measured_mtf_data[j].first << "\t"
              << measured_mtf_data[j].second << endl;
        }
      }
      *save_result = false;
    }
  }
}

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " config_file" << endl;
    return 1;
  }
  namedWindow("Image");
  waitKey(1);

  WaitQueue<Mat> image_queue(1), mtf_queue(1);
  bool save_results = false;
  bool get_new_roi = false;
  bool perform_mtf_analysis = false;
  bool average_mode = false;
  signal(SIGINT, CaptureSigint);

  // Initialize the model.
  mats::SimulationConfig sim_config;
  mats::DetectorParameters det_params;
  if (!mats::MatsInit(argv[1], &sim_config, &det_params, nullptr, nullptr)) {
    return 1;
  }

  // Create the telescope.
  mats::Telescope telescope(sim_config, 0, det_params);
  telescope.detector()->set_rows(512);
  telescope.detector()->set_cols(512);

  // Kick off MTF analysis. It has some upfront work to do, so it's best to
  // start it ASAP.
  thread mtf_thread(RunMTFAnalysis,
      &mtf_queue, &telescope, &average_mode, &keep_going, &save_results);

  // Connect to the detector.
  mats_io::SbigDetector detector;
  if (detector.has_error()) {
    keep_going = false;
    mtf_thread.join();
    return 1;
  }
  detector.set_error_callback(die);

  // Grab the size of the detector.
  uint16_t fr_width, fr_height;
  detector.GetSize(fr_width, fr_height);
  vector<uint16_t> roi{0, 0, fr_width, fr_height};

  // Set up the menu application.
  MenuApplication app;
  app.AddItem("Full Frame", "Full frame preview mode",
      [&app, &roi, fr_width, fr_height, &perform_mtf_analysis] () {
        roi[0] = 0;
        roi[1] = 0;
        roi[2] = fr_width;
        roi[3] = fr_height;
        perform_mtf_analysis = false;
      });

  app.AddItem("Set ROI", "Set the region of interest for analysis",
      [&app, &get_new_roi] () {
        get_new_roi = true;
      });

  app.AddItem("Toggle MTF Analysis",
              "Turn on/off slant-edge analysis. Set the ROI first.",
      [&app, &perform_mtf_analysis, &average_mode] () {
        perform_mtf_analysis = !perform_mtf_analysis;
        if (!perform_mtf_analysis) {
          average_mode = false;
        }
      });

  app.AddItem("Toggle MTF Averaging",
              "Turn on/off MTF averaging. Turn on MTF analysis first.",
      [perform_mtf_analysis, &average_mode] () {
        average_mode = !average_mode;
      });

  app.AddItem("Save", "Write out the next results",
      [&app, &save_results] () {
        app.PostStatusMessage("Saving...");
        save_results = true;
      });
        
  app.AddItem("Exit", "Close the camera",
      [/*&keep_going,*/ &app] () {
        keep_going = false;
        app.stop();
      });


  Mat scaled;
  int scaled_rows = 700;
  int scaled_cols = ((double) scaled_rows * fr_width) / fr_height;

  int image_num = 0;
  thread input_thread(InputThread, &app);
  image_thread = thread(mats_io::SbigAcquisitionThread,
      &image_queue, &detector, &roi, &FLAGS_exposure_time, &keep_going);
  while (keep_going) {
    // If we need a new ROI, clear out the preview window so the ROI window can
    // be seen.
    if (get_new_roi) {
      destroyWindow("Image");
      roi.clear();
      roi = {0, 0, fr_width, fr_height};

      unique_ptr<Mat> full_frame(image_queue.wait());
      while (full_frame->rows != fr_height || full_frame->cols != fr_width) {
        full_frame.reset(image_queue.wait());
      }
      roi = move(GetRoi(*full_frame));
      scaled_cols = ((double) scaled_rows * roi[2]) / roi[3];
      namedWindow("Image");
      get_new_roi = false;
    }

    // Grab an image from the current ROI and show a preview.
    Mat* raw_image_ptr = image_queue.wait_for(100);
    if (raw_image_ptr == nullptr) continue;

    unique_ptr<Mat> frame(raw_image_ptr);
    if (frame->rows != roi[3] || frame->cols != roi[2]) {
      app.PostStatusMessage("Incorrect Size");
      continue;
    }

    double min_val, max_val;
    minMaxLoc(*frame, &min_val, &max_val);
    app.PostStatusMessage(
        mats::StringPrintf("Min: %.0f, Max: %.0f", min_val, max_val));

    resize(*frame, scaled, Size(scaled_cols, scaled_rows));
    imshow("Image", ByteScale(scaled));

    // Push the image onto the MTF analysis queue.
    if (perform_mtf_analysis) {
      mtf_queue.push(frame.release());
    }
    waitKey(1);

    if (!perform_mtf_analysis && save_results) {
      imwrite(mats::StringPrintf("frame%d.png", image_num++), *frame);
      app.PostStatusMessage("Saved!");
      save_results = false;
    }
  }

  input_thread.join();
  cout << "Input thread done." << endl;

  image_thread.join();
  cout << "Image thread done." << endl;

  mtf_thread.join();
  cout << "MTF thread done." << endl;

  return 0;
}
