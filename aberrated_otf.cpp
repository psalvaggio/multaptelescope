// File Description
// Author: Philip Salvaggio

#include "mats.h"

#include <cstdlib>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>

#include <gflags/gflags.h>

using namespace std;
using namespace cv;
using namespace mats;

DEFINE_string(spectral_weighting, "", "Optional spectral weighting filename.");
DEFINE_bool(output_inverse_filter, false,
            "Whether to output the inverse filter.");
DEFINE_double(smoothness, 1e-3, "Smoothness Lagrange multiplier.");
DEFINE_bool(output_psf, false,
            "Whether to output the point spread function.");
DEFINE_int32(colormap, -1,
             "Which colormap to apply to MTF/PSF outputs. See OpenCV"
             "documentation for values.");

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (argc < 2) {
    cerr << "Usage: ./mats_main config_file [sim_id]" << endl;
    return 1;
  }

  SimulationConfig sim_config;
  DetectorParameters detector_params;
  if (!MatsInit(argv[1], &sim_config, &detector_params, NULL, NULL)) {
    return 1;
  }
  if (!sim_config.has_array_size()) sim_config.set_array_size(512);

  detector_params.set_array_rows(512);
  detector_params.set_array_cols(512);

  int sim_index = argc >= 3 ? LookupSimulationId(sim_config, atoi(argv[2])) : 0;

  Telescope telescope(sim_config, sim_index, detector_params);
  Mat otf;

  mainLog() << mats_io::PrintSimulation(telescope.simulation()) << endl;

  vector<PupilFunction> pupil;
  vector<double> tmp_wavelength{sim_config.reference_wavelength()};
  telescope.aperture()->GetPupilFunction(
      tmp_wavelength, 0, 0,
      sim_config.array_size(),
      sim_config.reference_wavelength(),
      &pupil);
  imwrite("mask.png", ByteScale(pupil[0].magnitude()));
  imwrite("wfe.png", ByteScale(pupil[0].phase()));

  vector<double> wavelengths, spectral_weighting;
  if (FLAGS_spectral_weighting == "") {
    wavelengths.push_back(sim_config.reference_wavelength());
    spectral_weighting.push_back(1);
  } else {
    vector<vector<double>> data;
    CHECK(mats_io::TextFileReader::Parse(
          ResolvePath(FLAGS_spectral_weighting), &data));
    wavelengths = move(data[0]);
    spectral_weighting = move(data[1]);
  }
  telescope.EffectiveOtf(wavelengths, spectral_weighting, 0, 0, &otf);

  Mat output_mtf = GammaScale(FFTShift(magnitude(otf)), 1/2.2);
  if (FLAGS_colormap >= 0) {
    output_mtf = ColorScale(output_mtf, FLAGS_colormap);
  }
  imwrite("mtf.png", output_mtf);

  if (FLAGS_output_inverse_filter) {
    ConstrainedLeastSquares cls;
    Mat inv_filter;
    cls.GetInverseFilter(otf, FLAGS_smoothness, &inv_filter);

    inv_filter = GammaScale(FFTShift(magnitude(inv_filter)), 1/2.2);
    if (FLAGS_colormap >= 0) {
      inv_filter = ColorScale(inv_filter, FLAGS_colormap);
    }
    imwrite("inv_filter.png", inv_filter);
  }

  if (FLAGS_output_psf) {
    Mat psf;
    dft(otf, psf, DFT_COMPLEX_OUTPUT);
    psf = GammaScale(FFTShift(magnitude(psf)), 1/2.2);
    if (FLAGS_colormap >= 0) {
      psf = ColorScale(psf, FLAGS_colormap);
    }
    imwrite("psf.png", psf);
  }

  return 0;
}
