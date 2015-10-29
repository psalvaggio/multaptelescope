// Program to restore a sparse aperture degraded image.
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
//DEFINE_string(output_inv_filter, "", "Optional output for the inverse filter");

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
  if (!MatsInit(ResolvePath(FLAGS_config_file), &sim_config, &detector_params,
                nullptr, nullptr)) {
    return 1;
  }

  // Find the correct simulation
  int sim_index = LookupSimulationId(sim_config, FLAGS_simulation_id);

  // Orient the telescope.
  sim_config.mutable_simulation(sim_index)->mutable_aperture_params()->
      set_rotation(FLAGS_orientation);

  // Read in the degraded image.
  Mat image = imread(ResolvePath(FLAGS_image), 0);
  if (!image.data) {
    cerr << "Could not read image file." << endl;
    return 1;
  }
  ConvertMatToDouble(image, image);

  // Read in the spectral weighting function. If there is not one, use the
  // detector QE spectrum.
  vector<double> wavelengths, illumination;
  if (file_exists(FLAGS_illumination)) {
    vector<vector<double>> sw_data;
    if (!mats_io::TextFileReader::Parse(FLAGS_illumination, &sw_data)) {
      cerr << "Could not read spectra weighting." << endl;
      return 1;
    }
    wavelengths.swap(sw_data[0]);
    illumination.swap(sw_data[1]);
  } else {
    int band_idx = FLAGS_band;
    for (int i = 0; i < detector_params.band(band_idx).wavelength_size(); i++) {
      wavelengths.push_back(detector_params.band(band_idx).wavelength(i));
      illumination.push_back(1);
    }
  }

  // Create the telescope.
  mats::Telescope telescope(sim_config, sim_index, detector_params);
  telescope.detector()->set_rows(image.rows);
  telescope.detector()->set_cols(image.cols);

  Mat_<double> restored;
  telescope.Restore(image, wavelengths, illumination, FLAGS_band, 
                    FLAGS_smoothness, &restored);

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
