// File Description
// Author: Philip Salvaggio

#include "base/scoped_ptr.h"
#include "base/opencv_utils.h"
#include "base/pupil_function.h"
#include "base/detector.h"
#include "base/simulation_config.pb.h"
#include "base/detector_parameters.pb.h"
#include "base/endian.h"
#include "base/str_utils.h"
#include "base/telescope.h"
#include "deconvolution/constrained_least_squares.h"
#include "io/input_reader.h"
#include "io/detector_reader.h"
#include "io/envi_image_reader.h"
#include "io/envi_image_header.pb.h"
#include "io/logging.h"
#include "optical_designs/cassegrain.h"
#include "optical_designs/triarm9.h"
#include "optical_designs/triarm9_parameters.pb.h"
#include "optical_designs/aperture_parameters.pb.h"

#include <cstdlib>
#include <string>
#include <fstream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "third_party/gnuplot-cpp/gnuplot_i.hpp"
#include <google/protobuf/text_format.h>

using namespace std;
using namespace cv;
using mats_io::Logging;

int main(int argc, char** argv) {
  string version = "1.0.0";

  if (argc < 2) {
    cerr << "Usage: ./mats_main base_dir" << endl;
    return 1;
  }

  // Parse the base directory from the command line.
  string base_dir(argv[1]);
  if (base_dir[base_dir.size() - 1] != '/') {
    base_dir.append("/");
  }

  // Initialize random number generators
  cv::theRNG().state = 74572548;
  srand(4279419);

  // Initialize logging.
  if (!Logging::Init(base_dir)) {
    cerr << "Could not open log files." << endl;
    return 1;
  }

  // Write the header to the log file
  mainLog() << "Multi-Aperture Telescope Simulation (MATS "
            << version << ") Main Log File" << endl << endl;

  // Initialize the simulation parameters.
  mats::SimulationConfig sim_config;
  sim_config.set_base_directory(base_dir);
  sim_config.set_altitude(1662546.0);
  sim_config.set_reference_wavelength(550e-9);

  string sim_file = base_dir + "input/simulations.txt";
  mats_io::InputReader reader;
  if (!reader.Read(sim_file, &sim_config)) {
    cerr << "Could not read simulations file." << endl;
    return 1;
  }

  mainLog() << "Configuration Parameters:" << endl
            << mats_io::PrintConfig(sim_config) << endl;

  // Initialize the detector parameters.
  string det_file = base_dir + "input/detector.txt";
  mats_io::DetectorReader det_reader;
  mats::DetectorParameters detector_params;
  if (!det_reader.Read(det_file, &detector_params)) {
    cerr << "Could not read detector file." << endl;
    return 1;
  }

  mainLog() << "Detector:" << endl << mats_io::PrintDetector(detector_params)
            << endl;

  // Read in the hyperspectral input image that will serve as the input to the
  // telescope.
  string img_file = base_dir + "input/input.img";
  mats_io::EnviImageReader envi_reader;
  vector<Mat> hyp_planes;
  mats_io::EnviImageHeader hyp_header;
  if (!envi_reader.Read(img_file, &hyp_header, &hyp_planes)) {
    cerr << "Could not read hyperspectral input file." << endl;
    return 1;
  }
  detector_params.set_array_rows(hyp_header.lines());
  detector_params.set_array_cols(hyp_header.samples());
  sim_config.set_array_size(std::max(detector_params.array_rows(),
                                     detector_params.array_cols()));
  


  // Convert the center wavelengths and FWHMs into meters.
  string wave_units;
  std::transform(hyp_header.wavelength_units().begin(),
                 hyp_header.wavelength_units().end(),
                 wave_units.begin(), ::tolower);

  // Wavenumbers [cm^-1]
  double wave_multiplier = 1e-6;
  bool is_wavenumber = wave_units == "wavenumbers";
  if (is_wavenumber) {
    wave_multiplier = 1e-2;
  } else if (wave_units == "microns" || wave_units == "micrometers") {
    wave_multiplier = 1e-6;
  } else if (wave_units == "nanometers") {
    wave_multiplier = 1e-9;
  } else {
    mainLog() << "WARNING: Wavelength units in ENVI header were missing or "
              << "an unrecognized unit. Assuming microns..." << endl;
  }

  vector<double> hyp_band_wavelengths;
  for (size_t i = 0; i < hyp_header.band_size(); i++) {
    double wave_val = hyp_header.band(i).center_wavelength();
    if (is_wavenumber) wave_val = 1 / wave_val;
    wave_val *= wave_multiplier;

    hyp_band_wavelengths.push_back(wave_val);
  }

  cout << "Ready to process " << sim_config.simulation_size()
       << " simulations" << endl;

  ApertureParameters ap;
  Triarm9Parameters* t9_params = ap.MutableExtension(triarm9_params);
  t9_params->set_subaperture_fill_factor(0.9);
  t9_params->set_s_to_d_ratio(1.03);

  for (size_t i = 0; i < 1; i++) {
    mainLog() << "Simulation " << (i+1) << " of "
              << sim_config.simulation_size() << endl
              << mats_io::PrintSimulation(sim_config.simulation(i)) << endl;

    mats::Telescope telescope(sim_config, i, ap, detector_params);

    vector<double> wavelengths;
    for (int i = 0; i < 51; i++) {
      wavelengths.push_back(400e-9 + 10e-9*i);
    }

    vector<Mat> output_image, otfs;
    telescope.Image(hyp_planes, hyp_band_wavelengths, &output_image, &otfs);

    ConstrainedLeastSquares cls;

    for (size_t i = 0; i < output_image.size(); i++) {
      //Mat deconvolved;
      //cls.Deconvolve(output_image[i], otfs[i], 1, &deconvolved);
      imshow(mats::StringPrintf("Output Image %lu", i),
             ByteScale(output_image[i]));
      waitKey(1000);
    }
  }

  waitKey(0);
  return 0;
}
