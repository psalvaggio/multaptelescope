// File Description
// Author: Philip Salvaggio

#include "mats.h"

#include <numeric>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <gflags/gflags.h>

DEFINE_string(config_file, "", "SimulationConfig filename.");
DEFINE_string(target_file, "", "Target image filename.");
DEFINE_double(orientation, 0, "Aperture orientation.");
DEFINE_double(full_well_frac, 0.8, "Fraction of the full-well capacity for the"
                                   " bright regions.");
DEFINE_string(output_prefix, "./", "Output directory + filename prefix");
DEFINE_string(illumination, "", "Illumination filename");
DEFINE_bool(parallelism, false, "Whether to use parallelism.");

using namespace std;
using namespace cv;
using namespace mats;

static const double h = 6.62606957e-34;  // [J*s]
static const double c = 299792458;  // [m/s]

void CorrectExposure(const Mat& image,
                     const vector<double>& wavelengths,
                     const vector<double>& illumination,
                     const Telescope& telescope,
                     vector<Mat>* output) {
  const Detector& det = *(telescope.detector());

  // Compute the central wavelength
  double central_wavelength = 0;
  for (size_t i = 0; i < wavelengths.size(); i++) {
    central_wavelength += wavelengths[i] * illumination[i];
  }

  // Do some radiometry to determine the correct exposure for the telescope.
  // Compute the desired irradiance [W/m^2] incident upon the detector.
  double target_electrons = FLAGS_full_well_frac * det.full_well_capacity();
  double eff_qe = det.GetEffectiveQE(wavelengths, illumination, 0);
  double target_photons = target_electrons / eff_qe;
  double int_time = telescope.simulation().integration_time();
  double target_flux = target_photons * h * c / central_wavelength;
  double target_irradiance = target_flux / (det.detector_area() * int_time);

  // Compute the effective G/# to convert the detector irradiance to
  // entrance-pupil reaching radiance.
  double eff_g_num = 0;
  for (size_t i = 0; i < illumination.size(); i++) {
    eff_g_num += illumination[i] * telescope.GNumber(wavelengths[i]);
  }

  // Construct the hyperspectral input at the proper light level
  output->clear();
  for (size_t i = 0; i < wavelengths.size(); i++) {
    output->push_back(image * target_irradiance * eff_g_num * illumination[i]);
  }
}

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  // Read in the input model parameters
  mats::SimulationConfig sim_config;
  mats::DetectorParameters detector_params;
  if (!MatsInit(ResolvePath(FLAGS_config_file),
                &sim_config,
                &detector_params)) {
    return 1;
  }

  // Read in the spectral weighting function for the binary target
  vector<vector<double>> data;
  if (!mats_io::TextFileReader::Parse(ResolvePath(FLAGS_illumination), &data)) {
    cerr << "Could not read illumination file." << endl;
    return 1;
  }
  const vector<double>& wavelengths = data[0];
  vector<double>& illumination = data[1];

  // Normalize the illumination function
  double total_weight = accumulate(begin(illumination), end(illumination), 0.);
  for (auto& tmp : illumination) tmp /= total_weight;

  // Read in the input image and peak normalize
  Mat image = imread(ResolvePath(FLAGS_target_file), 0);
  if (!image.data) {
    cerr << "Could not read image file." << endl;
    return 1;
  }
  flip(image, image, 1);
  image.convertTo(image, CV_64F);
  double im_max;
  minMaxLoc(image, NULL, &im_max);
  image /= im_max;

  // Set the size of the detector
  detector_params.set_array_rows(image.rows);
  detector_params.set_array_cols(image.cols);

  string output_path = ResolvePath(FLAGS_output_prefix);

  // Perform each simulation
  for (int i = 0; i < sim_config.simulation_size(); i++) {
    mainLog() << "Simulation " << (i+1) << " of "
              << sim_config.simulation_size() << endl
              << mats_io::PrintSimulation(sim_config.simulation(i));

    // Apply the command line flag for orientation
    sim_config.mutable_simulation(i)->
               mutable_aperture_params()->set_rotation(FLAGS_orientation);

    // Create the telescope
    mats::Telescope telescope(sim_config, i, detector_params);
    telescope.set_parallelism(FLAGS_parallelism);
    vector<Mat> spectral_radiance;
    CorrectExposure(image, wavelengths, illumination, telescope,
                    &spectral_radiance);

    // Image the input target
    vector<Mat> output_image;
    telescope.Image(spectral_radiance, wavelengths, &output_image);

    // For each output band, perfom inverse filtering and write output
    for (size_t band = 0; band < output_image.size(); band++) {
      // Compute the spectral weighting for the effective OTF
      vector<double> spectral_weighting, det_qe;
      telescope.detector()->GetQESpectrum(wavelengths, band, &det_qe);
      for (size_t i = 0; i < wavelengths.size(); i++) {
        spectral_weighting.push_back(det_qe[i] * illumination[i]);
      }

      // Convert the raw image to the 0-1 range
      Mat raw_image;
      output_image[band].convertTo(raw_image, CV_64F,
          1 / pow(2., telescope.detector()->bit_depth()));

      flip(raw_image, raw_image, 1);

      // Contrast scale and output the raw and processed image
      raw_image.convertTo(raw_image, CV_8U, 255);
      imwrite(StringPrintf("%ssim_%d_output_band_%d.png", output_path.c_str(),
                           i, band), raw_image);
    }
  }

  return 0;
}
