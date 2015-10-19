// File Description
// Author: Philip Salvaggio

#include "mats.h"

#include <cstdlib>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <string>

#include <gflags/gflags.h>

using namespace std;
using namespace cv;

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

  mats::SimulationConfig sim_config;
  mats::DetectorParameters detector_params;
  if (!mats::MatsInit(argv[1], &sim_config, &detector_params, NULL, NULL)) {
    return 1;
  }
  if (!sim_config.has_array_size()) sim_config.set_array_size(512);

  detector_params.set_array_rows(512);
  detector_params.set_array_cols(512);

  int sim_index = 0;
  if (argc >= 3) {
    int sim_id = atoi(argv[2]);
    for (int i = 0; i < sim_config.simulation_size(); i++) {
      if (sim_config.simulation(i).simulation_id() == sim_id) {
        sim_index = i;
        break;
      }
    }
  }

  mats::Telescope telescope(sim_config, sim_index, detector_params);
  Mat otf;

  mainLog() << mats_io::PrintAperture(telescope.aperture()->aperture_params())
            << endl;


  mats::PupilFunction pupil(sim_config.array_size(),
                            sim_config.reference_wavelength());
  telescope.aperture()->GetPupilFunction(
      sim_config.reference_wavelength(), 0, 0, &pupil);

  if (FLAGS_spectral_weighting == "") {
    vector<Mat> spectral_otf;
    telescope.ComputeOtf({sim_config.reference_wavelength()}, &spectral_otf);
    otf = spectral_otf[0];
  } else {
    vector<vector<double>> data;
    if (mats_io::TextFileReader::Parse(
          mats::ResolvePath(FLAGS_spectral_weighting), &data)) {
      telescope.ComputeEffectiveOtf(data[0], data[1], &otf);
    } else {
      cerr << "Could not read spectral weighting file." << endl;
    }
  }

  imwrite("mask.png", ByteScale(pupil.magnitude()));
  imwrite("wfe.png", ByteScale(pupil.phase()));

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
    Mat psf = GammaScale(FFTShift(pupil.PointSpreadFunction()), 1/2.2);
    if (FLAGS_colormap >= 0) {
      psf = ColorScale(psf, FLAGS_colormap);
    }
    imwrite("psf.png", psf);
  }

  return 0;
}
