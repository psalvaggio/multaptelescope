// File Description
// Author: Philip Salvaggio

#include "mats_init.h"

#include "base/filesystem.h"
#include "io/detector_reader.h"
#include "io/logging.h"
#include "io/envi_image_header.pb.h"
#include "io/envi_image_reader.h"
#include "io/input_reader.h"
#include <opencv/cv.h>

using namespace std;
using cv::Mat;

namespace mats {

bool MatsInit(const std::string& config_path,
              mats::SimulationConfig* sim_config,
              mats::DetectorParameters* detector_params,
              vector<Mat>* hyp_bands,
              mats_io::EnviImageHeader* hyp_header) {
  if (!sim_config || !detector_params) {
    cerr << "Invalid Pointer passed to mats::MatsInit()" << endl;
    return false;
  }


  string version = "1.0.0";

  // Initialize random number generators
  cv::theRNG().state = 74572548;
  srand(4279419);

  // Initialize logging.
  string config_file = config_path;
  if (is_dir(config_path)) {
    sim_config->set_base_directory(config_path);
    config_file += "input/simulations.txt";
    if (!mats_io::Logging::Init(config_path)) {
      cerr << "Could not open log files." << endl;
      return false;
    }
  } else if (!mats_io::Logging::Init()) {
    cerr << "Could not initialize logging." << endl;
    return false;
  }

  // Write the header to the log file
  mainLog() << "Multi-Aperture Telescope Simulation (MATS "
            << version << ") Main Log File" << endl << endl;

  // Initialize the simulation parameters.
  mats_io::InputReader reader;
  if (!reader.Read(config_file, sim_config)) {
    cerr << "Could not read simulations file." << endl;
    return false;
  }

  mainLog() << "Configuration Parameters:" << endl
            << mats_io::PrintConfig(*sim_config) << endl;

  // Initialize the detector parameters.
  mats_io::DetectorReader det_reader;
  if (!det_reader.Read(sim_config->detector_params_filename(),
                       detector_params)) {
    cerr << "Could not read detector file." << endl;
    return false;
  }

  mainLog() << "Detector:" << endl << mats_io::PrintDetector(*detector_params)
            << endl;

  // Read in the hyperspectral input image that will serve as the input to the
  // telescope.
  if (hyp_bands && hyp_header) {
    mats_io::EnviImageReader envi_reader;
    if (!envi_reader.Read(sim_config->input_image_filename(),
                          hyp_header, hyp_bands)) {
      cerr << "Could not read hyperspectral input file." << endl;
      return false;
    }

    // Set up the array sizes based on the size of the input image.
    detector_params->set_array_rows(hyp_header->lines());
    detector_params->set_array_cols(hyp_header->samples());
    sim_config->set_array_size(std::max(detector_params->array_rows(),
                                        detector_params->array_cols()));
  }

  return true;
}

}
