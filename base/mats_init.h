// File Description
// Author: Philip Salvaggio

#ifndef MATS_INIT_H
#define MATS_INIT_H

#include <string>
#include <vector>
#include <opencv/cv.h>

namespace mats_io {
class EnviImageHeader;
}

namespace mats {

class SimulationConfig;
class DetectorParameters;

// Perform initialization of the model.
//
// Parameters:
//  config_path     Either a path to the config file or the base directory. If
//                  this is a path to the file, then logging will be performed
//                  to stdout. If this is a path to a directory, then the
//                  config file should be input/simulations.txt and logging
//                  will be done to logs/main_log.txt.
//  sim_config      Output: SimulationConfig structure read from the file.
//  detector_params Output: Detector parameters
//  hyp_bands       Output: Bands of the hyperspectral input. NULL to not read.
//  hyp_header      Output: Metadata for the hyperspectral input. NULL to not
//                          read.
bool MatsInit(const std::string& base_directory,
              mats::SimulationConfig* sim_config,
              mats::DetectorParameters* detector_params,
              std::vector<cv::Mat>* hyp_bands,
              mats_io::EnviImageHeader* hyp_header);

}

#endif  // MATS_INIT_H
