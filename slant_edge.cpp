// Measure the MTF from a slant-edge region.
// Author: Philip Salvaggio

#include "mats.h"

#include <iostream>

#include <gflags/gflags.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

DEFINE_string(image_file, "", "REQUIRED: Image filename.");
DEFINE_string(config_file, "", "Optional: SimulationConfig filename for a "
                               "plot overlay.");
DEFINE_int32(simulation_id, -1, "Simulation ID in config_file "
                                "(defaults to first).");
DEFINE_double(orientation, 0, "Orientation of the aperture.");
DEFINE_bool(whole_image, false, "Use the entire image.");
DEFINE_bool(quiet, false, "Do not plot");
DEFINE_bool(output_esf, false, "Output the ESF instead of the MTF.");

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_image_file == "") {
    cerr << "Image filename is required." << endl;
    return 1;
  }

  SlantEdgeMtf mtf_measurer;
  Mat image = imread(mats::ResolvePath(FLAGS_image_file), 0);
  if (!image.data) {
    cerr << "Could not read image file." << endl;
    return 1;
  }

  // Extract the slant edge ROI from the image.
  Mat roi;
  if (FLAGS_whole_image) {
    roi = image;
  } else {
    auto bounds = move(GetRoi(image));
    image(Range(bounds[1], bounds[1] + bounds[3]),
          Range(bounds[0], bounds[0] + bounds[2])).copyTo(roi);
  }

  SlantEdgeMtf mtf_analyzer;
  if (FLAGS_output_esf) {
    double edge[3];
    if (mtf_analyzer.DetectEdge(roi, edge)) {
      int samples = mtf_analyzer.GetSamplesPerPixel(image, edge);
      vector<double> esf, esf_stddevs;
      mtf_analyzer.GenerateEsf(roi, edge, samples, &esf, &esf_stddevs);
      mtf_analyzer.SmoothEsf(&esf);
      for (size_t i = 0; i < esf.size(); i++) {
        cout << i / double(samples) << "\t" << esf[i] << endl;
      }
    }
    return 0;
  }

  // Perform the slant edge analysis on the image.
  double orientation;
  vector<double> mtf;
  mtf_analyzer.Analyze(roi, &orientation, &mtf);

  // Create the plot data and print out the MTF.
  vector<pair<double, double>> mtf_data;
  for (size_t i = 0; i < mtf.size(); i++) {
    double freq = i / (2. * (mtf.size() - 1));
    mtf_data.emplace_back(freq, mtf[i]);
    cout << freq << "\t" << mtf[i] << endl;
  }

  if (FLAGS_quiet) return 0;

  Gnuplot gp;
  gp << "set xlabel \"Spatial Frequency [cyc/pixel]\"\n"
     << "set ylabel \"MTF\"\n";

  if (FLAGS_config_file == "") {
     gp << "unset key\n"
        << "plot " << gp.file1d(mtf_data) << " w l lw 3\n"
        << endl;
  } else {
    mats::SimulationConfig sim_config;
    mats::DetectorParameters detector_params;
    if (!mats::MatsInit(mats::ResolvePath(FLAGS_config_file),
                        &sim_config,
                        &detector_params)) {
      return 1;
    }

    // Initialize the simulation parameters.
    sim_config.set_array_size(512);

    int sim_index = mats::LookupSimulationId(sim_config, FLAGS_simulation_id);
    mats::Telescope telescope(sim_config, sim_index, detector_params);
    telescope.detector()->set_rows(512);
    telescope.detector()->set_cols(512);

    // Set up the spectral resolution of the simulation.
    vector<vector<double>> raw_weighting;
    mats_io::TextFileReader::Parse(
        sim_config.spectral_weighting_filename(),
        &raw_weighting);
    const vector<double>& wavelengths(raw_weighting[0]),
                          spectral_weighting(raw_weighting[1]);

    double q = telescope.EffectiveQ(wavelengths, spectral_weighting);
    cerr << "F/#: " << telescope.FNumber() << endl;
    cerr << "Effective Q: " << q << endl;

    Mat_<complex<double>> theoretical_otf;
    telescope.EffectiveOtf(wavelengths, spectral_weighting, 0, 0,
                           &theoretical_otf);
    Mat theoretical_2d_mtf = magnitude(theoretical_otf);

    // Grab the radial profile of the OTF and convert to MTF.
    std::vector<double> theoretical_mtf;
    GetRadialProfile(FFTShift(theoretical_2d_mtf), 
                     FLAGS_orientation * M_PI / 180,
                     &theoretical_mtf);

    vector<pair<double, double>> t_mtf_data;
    for (size_t i = 0; i < theoretical_mtf.size(); i++) {
      double pix_freq = i / (2. * (theoretical_mtf.size() - 1));  // [cyc/pixel]
      t_mtf_data.emplace_back(pix_freq, theoretical_mtf[i]);
    }

    gp << "plot " << gp.file1d(mtf_data) << " w l lw 3 t \"Measured\", "
       << gp.file1d(t_mtf_data) << " w l lw 3 t \"Theoretical\"\n"
       << endl;
  }

  return 0;
}
