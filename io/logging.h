// File Description
// Author: Philip Salvaggio

#ifndef LOGGING_H
#define LOGGING_H

#include "base/simulation_config.pb.h"
#include "base/detector_parameters.pb.h"

#include <fstream>
#include <string>

namespace mats_io {

class Logging {
 public:
  static bool Init(const std::string& base_dir);

  static std::ofstream& Main();

 private:
  static std::ofstream main_logfile_;
  static bool inited_;

  Logging();
};

std::string PrintConfig(const mats::SimulationConfig& config);
std::string PrintSimulation(const mats::Simulation& simulation);
std::string PrintDetector(const mats::DetectorParameters& detector);

}

std::ofstream& mainLog();

#endif  // LOGGING_H
