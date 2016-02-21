// Program to restore a sparse aperture degraded image. To construct an inverse
// filter, the spectral weighting function of the system is needed. This is the
// product of the illumination spectrum and the system's spectral response. The
// system response is entirely defined by the config file, so only the
// illumination is left. This can be specified in one of three ways:
//
// 1. No illumination: Only the system spectral response will be used.
// 2. Illumination spectrum: A text file with the illumination spectrum will be
//                           used to restore the entire image. (.txt)
// 3. Hyperspectral image: A hyperspectral image (ENVI format) can be given
//                         to give a spatially-varying illumination.
// Author: Philip Salvaggio

#include "mats.h"

#include <fstream>
#include <string>

#include <gflags/gflags.h>
#include <google/protobuf/text_format.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
using namespace mats;

DEFINE_string(config_file, "", "REQUIRED: SimulationConfig filename.");
DEFINE_string(image, "", "REQUIRED: Image filename.");
DEFINE_double(orientation, 0, "Orientation of aperture, CCW from +x [deg]");
DEFINE_double(smoothness, 1e-2, "Smoothness for the inverse filter.");
DEFINE_int32(band, 0, "Band index for a multi-band system.");
DEFINE_int32(simulation_id, -1, "Simulation ID in config_file "
                                "(defaults to first).");
DEFINE_string(output, "restored.png", "Filename for the output.");
DEFINE_string(illumination, "", "Filename for the illumination function.");

void RestoreWithSystem(const cv::Mat& image,
                       const Telescope& telescope,
                       cv::Mat_<double>& restored) {
  const auto& det_params = telescope.detector()->det_params();

  vector<double> wavelengths, illumination;
  for (int i = 0; i < det_params.band(FLAGS_band).wavelength_size(); i++) {
    wavelengths.push_back(det_params.band(FLAGS_band).wavelength(i));
    illumination.push_back(1);
  }
  telescope.Restore(image, wavelengths, illumination, FLAGS_band, 
                    FLAGS_smoothness, &restored);
}


void RestoreWithSpectrum(const cv::Mat& image,
                         const Telescope& telescope,
                         cv::Mat_<double>& restored) {
  vector<vector<double>> sw_data;
  if (!mats_io::TextFileReader::Parse(FLAGS_illumination, &sw_data)) {
    cerr << "Could not read spectra weighting." << endl;
    exit(1);
  }
  telescope.Restore(image, sw_data[0], sw_data[1], FLAGS_band, 
                    FLAGS_smoothness, &restored);
}


void RestoreWithImage(const cv::Mat& image,
                      const Telescope& telescope,
                      cv::Mat_<double>& restored) {
  vector<double> wavelengths;
  vector<Mat> low_res_hyp;
  if (!mats_io::EnviImread(FLAGS_illumination, &wavelengths, &low_res_hyp)) {
    cerr << "Could not read spectra weighting." << endl;
    exit(1);
  }


  const int kRows = low_res_hyp[0].rows;
  const int kCols = low_res_hyp[0].cols;
  cout << "Hyperspectral weight resolution: " << kRows << " x " << kCols
       << " x " << low_res_hyp.size() << endl;
 
  telescope.Restore(image, wavelengths, low_res_hyp, FLAGS_band,
                    FLAGS_smoothness, &restored);
}


int main(int argc, char** argv) {
  google::SetUsageMessage("Restores an image that has been degraded with a "
                          "sparse aperture.");
  google::ParseCommandLineFlags(&argc, &argv, true);

  // Check required flags.
  if (FLAGS_config_file == "") {
    cerr << "--config_file is required." << endl;
    return 1;
  }
  if (FLAGS_image == "") {
    cerr << "--image is required." << endl;
    return 1;
  }

  // Read in the model parameters.
  SimulationConfig sim_config;
  DetectorParameters detector_params;
  if (!MatsInit(ResolvePath(FLAGS_config_file),
                &sim_config, &detector_params)) {
    return 1;
  }

  // Find the correct simulation
  int sim_index = LookupSimulationId(sim_config, FLAGS_simulation_id);

  // Orient the telescope.
  sim_config.mutable_simulation(sim_index)->mutable_aperture_params()->
      set_rotation(FLAGS_orientation);

  // Create the telescope.
  mats::Telescope telescope(sim_config, sim_index, detector_params);

  // Read in the degraded image.
  Mat image = imread(ResolvePath(FLAGS_image), 0);
  if (!image.data) {
    cerr << "Could not read image file." << endl;
    return 1;
  }
  ConvertMatToDouble(image, image);
  telescope.detector()->set_rows(image.rows);
  telescope.detector()->set_cols(image.cols);

  cout << "Mean Input " << mean(image) << endl;
  Mat_<double> restored;

  telescope.set_parallelism(true);
  if (file_exists(FLAGS_illumination)) {
    string ext = strtolower(Extension(FLAGS_illumination));
    if (ext == "txt") {
      RestoreWithSpectrum(image, telescope, restored);
    } else if (ext == "img") {
      telescope.set_parallelism(false);
      RestoreWithImage(image, telescope, restored);
    } else {
      cerr << "Error: Illumination file formats are .txt and .img." << endl;
      return 1;
    }
  } else {
    RestoreWithSystem(image, telescope, restored);
  }

  cout << "Mean Output " << mean(restored) << endl;

  // Inverse filtering tends to produce a lot of artifacts around the edges of
  // the image, so write out the center.
  Range row_range(restored.rows / 8., 7 * restored.rows / 8.),
        col_range(restored.cols / 8., 7 * restored.cols / 8.);
  Mat roi = restored(row_range, col_range);
  Mat output = roi * 255;
  output.convertTo(output, CV_8U);
  imwrite(mats::ResolvePath(FLAGS_output), output);

  return 0;
}
