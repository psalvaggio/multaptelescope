// File Description
// Author: Philip Salvaggio

#include "base/detector_parameters.pb.h"
#include "base/mats_init.h"
#include "base/opencv_utils.h"
#include "base/pupil_function.h"
#include "base/simulation_config.pb.h"
#include "base/telescope.h"
#include "io/envi_image_header.pb.h"
#include "io/logging.h"
#include "optical_designs/aperture.h"

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
    cerr << "Usage: ./mats_main base_dir [sim_id]" << endl;
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
  if (!mats::MatsInit(base_dir, &sim_config, &detector_params, NULL, NULL)) {
    return 1;
  }
  sim_config.set_array_size(512);

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

  cv::imwrite(base_dir + "mask.png", ByteScale(pupil.magnitude()));
  cv::imwrite(base_dir + "wfe.png", ByteScale(pupil.phase()));
  cv::imwrite(base_dir + "mtf.png", GammaScale(FFTShift(mtf), 1/2.2));

  return 0;
}
