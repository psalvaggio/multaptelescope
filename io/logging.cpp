// File Description
// Author: Philip Salvaggio

#include "logging.h"

#include <iostream>
#include <sstream>

using namespace std;
using namespace mats;

std::ostream& mainLog() {
  return mats_io::Logging::Main();
}

namespace mats_io {

bool Logging::using_stderr_ = false;
bool Logging::inited_ = false;
ofstream Logging::main_logfile_;

Logging::Logging() {}

bool Logging::Init() {
  inited_ = true;
  using_stderr_ = true;
  return true;
}

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

ostream& Logging::Main() {
  if (!inited_) {
    cerr << "Warning: Please call mats_io::Logging::Init() before trying "
         << "to log messages." << endl;
  }
  return using_stderr_ ? cerr : main_logfile_;
}

string PrintConfig(const SimulationConfig& config) {
  stringstream output;
  output << "Base directory: " << config.base_directory() << endl
         << "Altitude: " << config.altitude() << " [m]" << endl
         << "Reference Wavelength: " << config.reference_wavelength() << endl
         << "Array Size: " << config.array_size() << endl
         << "Number of simulations: " << config.simulation_size() << endl;

  return output.str();
}

string PrintSimulation(const Simulation& simulation) {
  stringstream output;
  
  output << "Simulation ID: " << simulation.simulation_id() << endl
         << "Integration Time: " << simulation.integration_time()
         << " [s]" << endl
         << "Ground Sample Distance: " << simulation.gsd()
         << " [m/pixel]" << endl;

  return output.str();
}

string PrintAperture(const ApertureParameters& ap_params) {
  stringstream output;

  output << "Type: " << ApertureParameters::ApertureType_Name(ap_params.type())
         << endl
         << "Encircled Diameter: " << ap_params.encircled_diameter() << endl
         << "Fill Factor: " << ap_params.fill_factor() << endl
         << "Aberrations: " << endl;

  for (int i = 0; i < ap_params.aberration_size(); i++) {
    const ZernikeCoefficient& ab(ap_params.aberration(i));

    output << "  " << ZernikeCoefficient::AberrationType_Name(ab.type())
           << ": " << ab.value() << endl;
  }

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

std::string PrintAberration(const mats::ZernikeCoefficient& aberration) {
  stringstream ss;
  ss << "Aberration ("
     << mats::ZernikeCoefficient::AberrationType_Name(aberration.type())
     << "): " << aberration.value();
  return ss.str();
}

}
