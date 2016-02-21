// File Description
// Author: Philip Salvaggio

#include "mats.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>

#include <google/protobuf/text_format.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
using mats_io::Logging;

int main(int argc, char** argv) {
  if (argc < 2) {
    cerr << "Usage: ./mats_main base_dir" << endl;
    return 1;
  }

  mats::SimulationConfig sim_config;
  mats::DetectorParameters detector_params;
  vector<Mat> hyp_planes;
  mats_io::EnviImageHeader hyp_header;
  if (!mats::MatsInit(argv[1], &sim_config, &detector_params)) return 1;

  // Read in the image and size our simulation around the image size.
  vector<double> wavelengths;
  mats_io::EnviImread(sim_config.input_image_filename(),
                      &wavelengths, &hyp_planes);
  detector_params.set_array_rows(hyp_planes[0].rows);
  detector_params.set_array_cols(hyp_planes[0].cols);
  sim_config.set_array_size(std::max(detector_params.array_rows(),
                                     detector_params.array_cols()));

  const double kGain = 1e-3;
  for (auto& tmp : hyp_planes) tmp *= kGain;

  cout << "Ready to process " << sim_config.simulation_size()
       << " simulations" << endl;

  vector<double> illumination;
  for (size_t i = 0; i < wavelengths.size(); i++) {
    illumination.push_back(sum(hyp_planes[i])[0]);
  }
  double hyp_sum = accumulate(begin(illumination), end(illumination), 0.);
  for (auto& tmp : illumination) tmp /= hyp_sum;

  for (int i = 0; i < sim_config.simulation_size(); i++) {
    mainLog() << "Simulation " << (i+1) << " of "
              << sim_config.simulation_size() << endl
              << mats_io::PrintSimulation(sim_config.simulation(i));

    mats::Telescope telescope(sim_config, i, detector_params);
    telescope.set_parallelism(true);

    mainLog() << "Focal Length: " << telescope.FocalLength() << " [m]" << endl;

    vector<Mat> output_image, output_ref;
    telescope.Image(hyp_planes, wavelengths, &output_image);

    vector<Mat> low_res_hyp;
    telescope.detector()->AggregateSignal(hyp_planes, wavelengths, false,
                                          &low_res_hyp);

    ConstrainedLeastSquares cls;

    cout << "Restoring..." << endl;
    const double kSmoothness = 1e-3;
    for (size_t band = 0; band < output_image.size(); band++) {
      vector<double> weighting;
      telescope.detector()->GetQESpectrum(wavelengths, band, &weighting);
      for (size_t j = 0; j < wavelengths.size(); j++) {
        weighting[j] *= illumination[j];
      }
      Mat_<complex<double>> otf;
      telescope.EffectiveOtf(wavelengths, weighting, 0, 0, &otf);

      Mat_<double> deconvolved;
      cls.Deconvolve(output_image[band], otf, kSmoothness, &deconvolved);
      imwrite(mats::StringPrintf("output_band_%d.png", band),
            ByteScale(output_image[band]));
      imwrite(mats::StringPrintf("processed_band_%d.png", band),
            ByteScale(deconvolved));
      imwrite(mats::StringPrintf("input_band_%d.png", band), 
            ByteScale(low_res_hyp[band]));
      imwrite(mats::StringPrintf("mtf_%d.png",
            (int)(detector_params.band(band).center_wavelength() * 1e9)), 
            GammaScale(FFTShift(magnitude(otf)), 1/2.2));
    }
  }

  return 0;
}
