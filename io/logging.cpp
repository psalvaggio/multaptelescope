// File Description
// Author: Philip Salvaggio

#include "logging.h"

#include <iostream>
#include <sstream>

using namespace std;
using mats::Simulation;
using mats::SimulationConfig;

std::ofstream& mainLog() {
  return mats_io::Logging::Main();
}

namespace mats_io {

bool Logging::inited_ = false;
ofstream Logging::main_logfile_;

Logging::Logging() {}

bool Logging::Init(const string& base_dir) {
  if (inited_) return true;

  string fname = base_dir + "logs/main_log.txt";
  main_logfile_.open(fname.c_str());
  if (!main_logfile_.is_open()) {
    return false;
  }

  inited_ = true;
  return true;
}

ofstream& Logging::Main() {
  if (!inited_) {
    cerr << "Warning: Please call mats_io::Logging::Init() before trying "
         << "to log messages." << endl;
  }
  return main_logfile_;
}

string PrintConfig(const SimulationConfig& config) {
  stringstream output;
  output << "Base directory: " << config.base_directory() << endl
         << "Altitude: " << config.altitude() << " [m]" << endl
         << "Number of simulations: " << config.simulation_size() << endl;

  return output.str();
}

string PrintSimulation(const Simulation& simulation) {
  stringstream output;
  
  output << "Simulation ID: " << simulation.simulation_id() << endl
         << "Aperture Type: ";

  int ap_type = simulation.aperture_type();
  if (ap_type == Simulation::HEX18) {
    output << "HEX18";
  } else if (ap_type == Simulation::TRIARM9) {
    output << "TRIARM9";
  } else if (ap_type == Simulation::CASSEGRAIN) {
    output << "CASSEGRAIN";
  }

  output << endl 
         << "Wavefront Error Knowledge Used in Reconstruction: ";

  int wfe_knowledge = simulation.wfe_knowledge();
  if (wfe_knowledge == Simulation::HIGH) {
    output << "High";
  } else if (wfe_knowledge == Simulation::MEDIUM) {
    output << "Medium";
  } else if (wfe_knowledge == Simulation::LOW) {
    output << "Low";
  } else if (wfe_knowledge == Simulation::NONE) {
    output << "None";
  }

  output << endl << "Integration Time: " << simulation.integration_time()
         << " [s]" << endl
         << "Encircled Diameter of Aperture: "
         << simulation.encircled_diameter() << " [m]" << endl
         << "Fill Factor: " << simulation.fill_factor() * 100 << "%" << endl
         << "Ground Sample Distance: " << simulation.gsd()
         << " [m/pixel]" << endl;

  return output.str();
}

string PrintDetector(const mats::DetectorParameters& detector) {
  stringstream output;

  output << "Pixel Pitch: " << detector.pixel_pitch() << " [m]" << endl
         << "Temperature: " << detector.temperature() << " [K]" << endl
         << "Dark Current Doubling Temperature: "
           << detector.darkcurr_doubling_temp() << " [K]" << endl
         << "Output Sensitivity: " << detector.output_sensitivity() << " [K]"
           << endl
         << "Read Noise RMS: " << detector.read_rms() << " [rms electrons]"
           << endl
         << "Readout Time: " << detector.readout_time() << " [s]" << endl;

  return output.str();
}

}
