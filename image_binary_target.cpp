// File Description
// Author: Philip Salvaggio

#include "mats.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <gflags/gflags.h>

DEFINE_string(config_file, "", "SimulationConfig filename.");
DEFINE_string(target_file, "", "Target image filename.");
DEFINE_double(smoothness, 1e-2, "Inverse filtering smoothness.");
DEFINE_double(orientation, 0, "Aperture orientation.");
DEFINE_bool(parallelism, false, "Specify for parallel computation.");

using namespace std;
using namespace cv;
using mats_io::Logging;

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  mats::SimulationConfig sim_config;
  mats::DetectorParameters detector_params;
  if (!mats::MatsInit(FLAGS_config_file,
                      &sim_config,
                      &detector_params,
                      nullptr, nullptr)) {
    return 1;
  }

  vector<vector<double>> data;
  if (!mats_io::TextFileReader::Parse(sim_config.spectral_weighting_filename(),
                                      &data)) {
    cerr << "Could not read spectral weighting file." << endl;
    return 1;
  }
  const vector<double>& wavelengths = data[0];
  const vector<double>& illumination = data[1];

  Mat image = imread(FLAGS_target_file, 0);
  if (!image.data) {
    cerr << "Could not read image file." << endl;
    return 1;
  }
  image.convertTo(image, CV_64F);
  double im_max;
  minMaxLoc(image, NULL, &im_max);
  image /= im_max;

  vector<Mat> hyp_image;
  for (size_t i = 0; i < wavelengths.size(); i++) {
    hyp_image.push_back(image * illumination[i]);
  }

  for (int i = 0; i < sim_config.simulation_size(); i++) {
    mainLog() << "Simulation " << (i+1) << " of "
              << sim_config.simulation_size() << endl
              << mats_io::PrintSimulation(sim_config.simulation(i))
              << mats_io::PrintAperture(
                  sim_config.simulation(i).aperture_params());

    sim_config.mutable_simulation(i)->
               mutable_aperture_params()->set_rotation(FLAGS_orientation);
    mats::Telescope telescope(sim_config, i, detector_params);
    telescope.set_parallelism(FLAGS_parallelism);

    vector<Mat> output_image, output_ref;
    telescope.Image(hyp_image, wavelengths, &output_image);

    vector<Mat> low_res_hyp;
    telescope.detector()->AggregateSignal(hyp_image, wavelengths,
                                          false, &low_res_hyp);

    ConstrainedLeastSquares cls;

    cout << "Restoring..." << endl;
    for (size_t band = 0; band < output_image.size(); band++) {
      vector<double> spectral_weighting, det_qe;
      telescope.detector()->GetQESpectrum(wavelengths, band, &det_qe);
      for (size_t i = 0; i < wavelengths.size(); i++) {
        spectral_weighting.push_back(det_qe[i] * illumination[i]);
      }

      Mat deconvolved;
      Mat eff_otf;
      telescope.ComputeEffectiveOtf(wavelengths, spectral_weighting, &eff_otf);
      resize(eff_otf, eff_otf, output_image[band].size());

      cls.Deconvolve(output_image[band], eff_otf, FLAGS_smoothness,
          &deconvolved);

      Mat inv_filter;
      cls.GetInverseFilter(eff_otf, FLAGS_smoothness, &inv_filter);

      imwrite(mats::StringPrintf("sim_%d_output_band_%d.png", i, band),
              ByteScale(output_image[band]));
      imwrite(mats::StringPrintf("sim_%d_processed_band_%d.png", i, band),
              ByteScale(deconvolved));
      imwrite(mats::StringPrintf("sim_%d_input_band_%d.png", i, band), 
              ByteScale(low_res_hyp[band]));
      imwrite(mats::StringPrintf("sim_%d_mtf_%d.png", i, band),
              GammaScale(FFTShift(magnitude(eff_otf)), 1/2.2));
      imwrite(mats::StringPrintf("sim_%d_inv_filter_%d.png", i, band),
              GammaScale(FFTShift(magnitude(inv_filter)), 1/2.2));
    }
  }

  return 0;
}
