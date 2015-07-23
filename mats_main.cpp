// File Description
// Author: Philip Salvaggio

#include "mats.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <google/protobuf/text_format.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>

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
  if (!mats::MatsInit(argv[1],
                      &sim_config,
                      &detector_params,
                      &hyp_planes,
                      &hyp_header)) {
    return 1;
  }

  // Convert the center wavelengths and FWHMs into meters.
  bool is_wavenumber = false;
  double wave_multiplier = mats_io::EnviImageReader::GetWavelengthMultiplier(
      hyp_header.wavelength_units(), &is_wavenumber);

  const double kGain = 10;

  vector<double> hyp_band_wavelengths;
  for (int i = 0; i < hyp_header.band_size(); i++) {
    hyp_planes[i].convertTo(hyp_planes[i], CV_64F);

    hyp_planes[i] *= 1e4;  // [W/m^2/sr micron^-1]
    hyp_planes[i] *= kGain;  // [W/m^2/sr micron^-1]

    double wave_val = hyp_header.band(i).center_wavelength();
    if (is_wavenumber) wave_val = 1 / wave_val;
    wave_val *= wave_multiplier;

    hyp_band_wavelengths.push_back(wave_val);
  }

  mainLog() << "Read input hyperspectral image with "
            << hyp_planes.size() << " bands." << endl;
  mainLog() << mats_io::PrintEnviHeader(hyp_header);

  cout << "Ready to process " << sim_config.simulation_size()
       << " simulations" << endl;

  for (int i = 0; i < sim_config.simulation_size(); i++) {
    mainLog() << "Simulation " << (i+1) << " of "
              << sim_config.simulation_size() << endl
              << mats_io::PrintSimulation(sim_config.simulation(i))
              << mats_io::PrintAperture(
                  sim_config.simulation(i).aperture_params());

    mats::Telescope telescope(sim_config, i, detector_params);

    mainLog() << "Focal Length: " << telescope.FocalLength() << " [m]" << endl;

    vector<Mat> output_image, otfs, output_ref, ref_otfs;
    telescope.Image(hyp_planes, hyp_band_wavelengths, &output_image, &otfs);

    vector<Mat> low_res_hyp;
    telescope.detector()->AggregateSignal(hyp_planes, hyp_band_wavelengths,
                                          false, &low_res_hyp);

    ConstrainedLeastSquares cls;

    cout << "Restoring..." << endl;
    const double kSmoothness = 1e-3;
    for (size_t band = 0; band < output_image.size(); band++) {
      Mat deconvolved;
      cls.Deconvolve(output_image[band], otfs[band], kSmoothness, &deconvolved);
      imwrite(mats::StringPrintf("output_band_%d.png", band),
            ByteScale(output_image[band]));
      imwrite(mats::StringPrintf("processed_band_%d.png", band),
            ByteScale(deconvolved));
      imwrite(mats::StringPrintf("input_band_%d.png", band), 
            ByteScale(low_res_hyp[band]));
      imwrite(mats::StringPrintf("mtf_%d.png",
            (int)(detector_params.band(band).center_wavelength() * 1e9)), 
            GammaScale(FFTShift(magnitude(otfs[band])), 1/2.2));
    }
  }

  waitKey(0);
  return 0;
}
