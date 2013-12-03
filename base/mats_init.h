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

bool MatsInit(const std::string& base_directory,
              mats::SimulationConfig* sim_config,
              mats::DetectorParameters* detector_params,
              std::vector<cv::Mat>* hyp_bands,
              mats_io::EnviImageHeader* hyp_header);

}

#endif  // MATS_INIT_H
