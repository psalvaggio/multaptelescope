// File Description
// Author: Philip Salvaggio

#include "input_reader.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <utility>

using mats::SimulationConfig;
using mats::Simulation;

namespace mats_io {

InputReader::InputReader() {}
InputReader::~InputReader() {}

bool InputReader::Read(const std::string& filename,
                       SimulationConfig* simulation) {
  if (simulation == NULL) return false;

  std::ifstream ifs;

  ifs.open(filename.c_str());
  if (!ifs.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return false;
  }

  std::string line;
  while (!std::getline(ifs, line).eof()) {
    Simulation* tmp_sim = simulation->add_simulation();
    parseLine(line, tmp_sim);
  }

  return true;
}

void InputReader::parseLine(const std::string& line,
                            Simulation* input_params) {
  if (input_params == NULL) return;

  int simulation_id, aperture_type, wfe_knowledge, num_output_bands;
  float ptt_opd_rms, ho_opd_rms, integration_time, encircled_diameter,
        fill_factor, gsd;
  int status = sscanf(line.c_str(), "%d, %d, %f, %f, %d, %f, %f, %f, %f, %d",
         &simulation_id, &aperture_type, &ptt_opd_rms, &ho_opd_rms,
         &wfe_knowledge, &integration_time, &encircled_diameter,
         &fill_factor, &gsd, &num_output_bands);

  if (status != 10) {
    std::cerr << "Warning: Faulty input line detected" << std::endl;
  }

  gsd *= 0.0254; // in/pix -> m/pix
  integration_time *= 1e-6; // microseconds -> seconds
  
  input_params->set_simulation_id(simulation_id);

  input_params->set_aperture_type((Simulation::ApertureType)aperture_type);

  input_params->set_ptt_opd_rms(ptt_opd_rms);
  input_params->set_ho_opd_rms(ho_opd_rms);

  if (wfe_knowledge == Simulation::LOW) {
    input_params->set_wfe_knowledge(Simulation::LOW);
  } else if (wfe_knowledge == Simulation::MEDIUM) {
    input_params->set_wfe_knowledge(Simulation::MEDIUM);
  } else if (wfe_knowledge == Simulation::HIGH) {
    input_params->set_wfe_knowledge(Simulation::HIGH);
  } else {
    input_params->set_wfe_knowledge(Simulation::NONE);
  }

  input_params->set_integration_time(integration_time);
  input_params->set_encircled_diameter(encircled_diameter);
  input_params->set_fill_factor(fill_factor);
  input_params->set_gsd(gsd);
}

}
