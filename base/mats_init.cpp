// File Description
// Author: Philip Salvaggio

#include "mats_init.h"

#include "base/filesystem.h"
#include "io/logging.h"
#include "io/envi_image_header.pb.h"
#include "io/envi_image_reader.h"
#include "io/protobuf_reader.h"

#include <iostream>

using namespace std;
using cv::Mat;

namespace mats {

bool MatsInit(const std::string& config_path,
              mats::SimulationConfig* sim_config,
              mats::DetectorParameters* detector_params) {
  if (!sim_config) {
    cerr << "Invalid Pointer passed to mats::MatsInit()" << endl;
    return false;
  }

  string version = "1.0.0";

  // Initialize random number generators
  cv::theRNG().state = cv::getTickCount();
  srand(cv::getTickCount());

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
  if (!mats_io::ProtobufReader::Read(config_file, sim_config)) {
    cerr << "Could not read simulations file." << endl;
    return false;
  }

  mainLog() << "Configuration Parameters:" << endl
            << mats_io::PrintConfig(*sim_config) << endl;

  // Initialize the detector parameters.
  if (detector_params) {
    if (!mats_io::ProtobufReader::Read(sim_config->detector_params_filename(),
                                       detector_params)) {
      cerr << "Could not read detector file." << endl;
      return false;
    }

    mainLog() << "Detector:" << endl << mats_io::PrintDetector(*detector_params)
              << endl;
  }

  return true;
}

int LookupSimulationId(const mats::SimulationConfig& sim_config,
                       int simulation_id) {
  for (int i = 0; i < sim_config.simulation_size(); i++) {
    if (sim_config.simulation(i).simulation_id() == simulation_id) {
      return i;
    }
  }
  return 0;
}

}
