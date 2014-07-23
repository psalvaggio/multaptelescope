// File Description
// Author: Philip Salvaggio

#include "base/aberration_factory.h"
#include "base/aperture_parameters.pb.h"
#include "base/detector.h"
#include "base/detector_parameters.pb.h"
#include "base/mats_init.h"
#include "base/opencv_utils.h"
#include "base/photon_noise.h"
#include "base/pupil_function.h"
#include "base/simulation_config.pb.h"
#include "base/str_utils.h"
#include "base/telescope.h"
#include "deconvolution/constrained_least_squares.h"
#include "io/logging.h"
#include "io/envi_image_header.pb.h"
#include "io/envi_image_reader.h"
#include "io/hdf5_reader.h"
#include "optical_designs/cassegrain.h"
#include "optical_designs/triarm9.h"
#include "optical_designs/triarm9_parameters.pb.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <google/protobuf/text_format.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <string>

using namespace std;
using namespace cv;
using mats_io::Logging;

int main(int argc, char** argv) {
  if (argc < 2) {
    cerr << "Usage: ./mats_main base_dir" << endl;
    return 1;
  }

  // Parse the base directory from the command line.
  string base_dir(argv[1]);
  if (base_dir[base_dir.size() - 1] != '/') {
    base_dir.append("/");
  }

  mats::SimulationConfig sim_config;
  mats::DetectorParameters detector_params;
  vector<Mat> hyp_planes;
  mats_io::EnviImageHeader hyp_header;
  if (!mats::MatsInit(base_dir,
                      &sim_config,
                      &detector_params,
                      &hyp_planes,
                      &hyp_header)) {
    return 1;
  }

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

  for (size_t i = 0; i < 1; i++) {
    mainLog() << "Simulation " << (i+1) << " of "
              << sim_config.simulation_size() << endl
              << mats_io::PrintSimulation(sim_config.simulation(i))
              << mats_io::PrintAperture(
                  sim_config.simulation(i).aperture_params());

    mats::Telescope telescope(sim_config, i, detector_params);

    mainLog() << "Focal Length: " << telescope.FocalLength() << " [m]" << endl;

    Aperture* ap = telescope.aperture();
    
    cv::imwrite(base_dir + "mask.png", ByteScale(ap->GetApertureMask()));

    Mat wfe = telescope.aperture()->GetWavefrontError();
    cv::imwrite(base_dir + "wfe.png", ByteScale(wfe));
    
    Mat wfe_est = telescope.aperture()->GetWavefrontErrorEstimate();
    cv::imwrite(base_dir + "wfe_est.png", ByteScale(wfe_est));

    cout << "Imaging..." << endl;
    vector<Mat> output_image, otfs, output_ref, ref_otfs;
    telescope.Image(hyp_planes, hyp_band_wavelengths, &output_image, &otfs);

    vector<Mat> low_res_hyp;
    telescope.detector()->AggregateSignal(hyp_planes, hyp_band_wavelengths,
                                          false, &low_res_hyp);

    ConstrainedLeastSquares cls;
    //namedWindow("Input Image");
    //namedWindow("Output Image");
    //moveWindow("Input Image", 0, 0);
    //moveWindow("Output Image", 500, 0);

    cout << "Restoring..." << endl;
    const double kSmoothness = 1e-3;
    for (size_t band = 0; band < output_image.size(); band++) {
      Mat deconvolved;
      cls.Deconvolve(output_image[band], otfs[band], kSmoothness, &deconvolved);
      imwrite(mats::StringPrintf(
            "%soutput_band_%d.png", base_dir.c_str(), band),
            ByteScale(output_image[band]));
      imwrite(mats::StringPrintf(
            "%sprocessed_band_%d.png", base_dir.c_str(), band),
            ByteScale(deconvolved));
      imwrite(mats::StringPrintf(
            "%sinput_band_%d.png", base_dir.c_str(), band), 
            ByteScale(low_res_hyp[band]));
      imwrite(mats::StringPrintf(
            "%smtf_%d.png", base_dir.c_str(), (int)(detector_params.band(band).center_wavelength() * 1e9)), 
            GammaScale(FFTShift(magnitude(otfs[band])), 1/2.2));
    }
  }

  waitKey(0);
  return 0;
}
