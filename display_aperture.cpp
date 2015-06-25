// Displays an aperture specified in a SimulationConfig protobuf.
// Author: Philip Salvaggio

#include "mats.h"

#include <iostream>
#include <opencv/highgui.h>

using namespace std;

int main(int argc, char** argv) {
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " sim_file [sim_id]" << endl;
    return 1;
  }

  mats::SimulationConfig sim_config;
  if (!mats::MatsInit(argv[1], &sim_config, nullptr, nullptr, nullptr)) {
    cerr << "Could not read simulation file." << endl;
    return 1;
  }

  int sim_index = 0;
  if (argc >= 3) {
    int sim_id = atoi(argv[2]);
    for (int i = 0; i < sim_config.simulation_size(); i++) {
      if (sim_config.simulation(i).simulation_id() == sim_id) {
        sim_index = i;
        break;
      }
    }
  }

  if (sim_index >= sim_config.simulation_size()) {
    cerr << "Not enough simulations present in the SimulationConfig." << endl;
    return 1;
  }

  unique_ptr<Aperture> ap(ApertureFactory::Create(sim_config, sim_index));
  if (ap.get() == nullptr) {
    cerr << "Invalid aperture configuration" << endl;
    return 1;
  }

  cv::imshow("Aperture Mask", ByteScale(ap->GetApertureMask()));
  cv::imshow("Wavefront Error", ByteScale(ap->GetWavefrontError()));
  cv::waitKey(0);

  return 0;
}
