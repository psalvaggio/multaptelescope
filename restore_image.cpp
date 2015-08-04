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
  int sim_index = 0;
  if (FLAGS_simulation_id >= 0) {
    for (int i = 0; i < sim_config.simulation_size(); i++) {
      if (sim_config.simulation(i).simulation_id() == FLAGS_simulation_id) {
        sim_index = i;
        break;
      }
    }
  }

  // Orient the telescope.
  sim_config.mutable_simulation(sim_index)->mutable_aperture_params()->
      set_rotation(FLAGS_orientation);

  // Read in the degraded image.
  Mat image = imread(ResolvePath(FLAGS_image), 0);
  if (!image.data) {
    cerr << "Could not read image file." << endl;
    return 1;
  }
  image.convertTo(image, CV_64F);

  // Read in the spectral weighting function. If there is not one, use the
  // detector QE spectrum.
  vector<double> wavelengths, spectral_weighting;
  if (sim_config.has_spectral_weighting_filename()) {
    vector<vector<double>> sw_data;
    if (!mats_io::TextFileReader::Parse(
          sim_config.spectral_weighting_filename(), &sw_data)) {
      cerr << "Could not read spectra weighting." << endl;
      return 1;
    }
    wavelengths.swap(sw_data[0]);
    spectral_weighting.swap(sw_data[1]);
  } else {
    int band_idx = FLAGS_band;
    for (int i = 0; i < detector_params.band(band_idx).wavelength_size(); i++) {
      wavelengths.push_back(detector_params.band(band_idx).wavelength(i));
      spectral_weighting.push_back(
          detector_params.band(band_idx).quantum_efficiency(i));
    }
  }

  // Create the telescope.
  mats::Telescope telescope(sim_config, sim_index, detector_params);
  telescope.detector()->set_rows(image.rows);
  telescope.detector()->set_cols(image.cols);

  // Get the effective OTF.
  Mat otf;
  telescope.ComputeEffectiveOtf(wavelengths, spectral_weighting, &otf);

  // Restore the image.
  Mat restored;
  ConstrainedLeastSquares cls;
  cls.Deconvolve(image, otf, FLAGS_smoothness, &restored);

  // Write out the invese filters.
  Mat inv_filter;
  cls.GetInverseFilter(otf, FLAGS_smoothness, &inv_filter);
  imwrite("inv_filter.png", ColorScale(GammaScale(FFTShift(
      magnitude(inv_filter)), 1/2.2), COLORMAP_JET));

  // Inverse filtering tends to produce a lot of artifacts around the edges of
  // the image, so write out the center.
  Range row_range(restored.rows / 4., 3 * restored.rows / 4.),
        col_range(restored.cols / 4., 3 * restored.cols / 4.);
  Mat roi = restored(row_range, col_range);
  imwrite("restored.png", ByteScale(roi));

  return 0;
}
