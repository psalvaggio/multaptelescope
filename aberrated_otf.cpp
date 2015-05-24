// File Description
// Author: Philip Salvaggio

#include "mats.h"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <string>

#include "base/zernike_aberrations.h"

using namespace std;
using namespace cv;
using mats_io::Logging;

int main(int argc, char** argv) {
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

  Mat wfe = telescope.aperture()->GetWavefrontError();

  mats::PupilFunction pupil;
  telescope.aperture()->GetPupilFunction(wfe,
      sim_config.reference_wavelength(), &pupil);

  Mat mtf = pupil.ModulationTransferFunction();

  cv::imwrite("mask.png", ByteScale(pupil.magnitude()));
  cv::imwrite("wfe.png", ByteScale(pupil.phase()));
  cv::imwrite("mtf.png", GammaScale(FFTShift(mtf), 1/2.2));

  return 0;
}
