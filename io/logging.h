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
  // Initialize and log to standard error.
  static bool Init();

  // Initialize and log to a log file.
  static bool Init(const std::string& base_dir);

  static std::ostream& Main();

 private:
  static std::ofstream main_logfile_;
  static bool inited_;
  static bool using_stderr_;

  Logging();
};

std::string PrintConfig(const mats::SimulationConfig& config);
std::string PrintSimulation(const mats::Simulation& simulation);
std::string PrintAperture(const mats::ApertureParameters& detector);
std::string PrintDetector(const mats::DetectorParameters& detector);

}

std::ostream& mainLog();

#endif  // LOGGING_H
