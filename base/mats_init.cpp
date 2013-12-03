// File Description
// Author: Philip Salvaggio

#include "mats_init.h"

#include "io/detector_reader.h"
#include "io/logging.h"
#include "io/envi_image_header.pb.h"
#include "io/envi_image_reader.h"
#include "io/input_reader.h"
#include <opencv/cv.h>

using namespace std;
using cv::Mat;

namespace mats {

bool MatsInit(const std::string& base_directory,
              mats::SimulationConfig* sim_config,
              mats::DetectorParameters* detector_params,
              vector<Mat>* hyp_bands,
              mats_io::EnviImageHeader* hyp_header) {
  if (!sim_config || !detector_params || !hyp_bands || !hyp_header) {
    cerr << "Invalid Pointer passed to mats::MatsInit()" << endl;
    return false;
  }


  string version = "1.0.0";

  // Initialize random number generators
  cv::theRNG().state = 74572548;
  srand(4279419);

  // Initialize logging.
  if (!mats_io::Logging::Init(base_directory)) {
    cerr << "Could not open log files." << endl;
    return false;
  }

  // Write the header to the log file
  mainLog() << "Multi-Aperture Telescope Simulation (MATS "
            << version << ") Main Log File" << endl << endl;

  // Initialize the simulation parameters.
  sim_config->set_base_directory(base_directory);
  string sim_file = base_directory + "input/simulations.txt";
  mats_io::InputReader reader;
  if (!reader.Read(sim_file, sim_config)) {
    cerr << "Could not read simulations file." << endl;
    return false;
  }

  mainLog() << "Configuration Parameters:" << endl
            << mats_io::PrintConfig(*sim_config) << endl;

  // Initialize the detector parameters.
  string det_file = base_directory + "input/detector.txt";
  mats_io::DetectorReader det_reader;
  if (!det_reader.Read(det_file, detector_params)) {
    cerr << "Could not read detector file." << endl;
    return false;
  }

  mainLog() << "Detector:" << endl << mats_io::PrintDetector(*detector_params)
            << endl;

  // Read in the hyperspectral input image that will serve as the input to the
  // telescope.
  string img_file = base_directory + "input/input.img";
  mats_io::EnviImageReader envi_reader;
  if (!envi_reader.Read(img_file, hyp_header, hyp_bands)) {
    cerr << "Could not read hyperspectral input file." << endl;
    return false;
  }

  // Set up the array sizes based on the size of the input image.
  detector_params->set_array_rows(hyp_header->lines());
  detector_params->set_array_cols(hyp_header->samples());
  sim_config->set_array_size(std::max(detector_params->array_rows(),
                                      detector_params->array_cols()));

  return true;
}

}
