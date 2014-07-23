// File Description
// Author: Philip Salvaggio

#include "triarm3.h"

#include "base/aberration_factory.h"
#include "base/aperture_parameters.pb.h"
#include "base/simulation_config.pb.h"
#include "io/logging.h"
#include "optical_designs/cassegrain.h"
#include "optical_designs/triarm3_parameters.pb.h"

#include <algorithm>
#include <fstream>
#include <opencv/highgui.h>

using namespace cv;
using namespace std;
using mats::Simulation;

Triarm3::Triarm3(const mats::SimulationConfig& params, int sim_index)
    : Aperture(params, sim_index),
      diameter_(aperture_params().encircled_diameter()),
      subap_diameter_(),
      subap_secondary_diameter_(),
      subaperture_offsets_(),
      mask_(),
      opd_(),
      opd_est_() {
  // Copy over the Triarm3-specific parameters.
  triarm3_params_ = this->aperture_params().GetExtension(triarm3_params);

  const int kSize = this->params().array_size();
  const int kHalfSize = kSize / 2;

  // The fill factor of the overall aperture and each of the individual
  // Cassegrain subapertures.
  double fill_factor = aperture_params().fill_factor();
  double subap_fill_factor = triarm3_params_.subaperture_fill_factor();

  const double kTargetRadius = kHalfSize;  // [pixels]

  // Compute the enclosed area and the area of each subapertures [pixels^2]
  double area_enclosed = M_PI * kTargetRadius * kTargetRadius;
  double area_subap = area_enclosed * fill_factor / kNumApertures;

  // Solve for the radius of the subapertures [pixels]
  subap_diameter_ = 2 * sqrt(area_subap / (subap_fill_factor * M_PI));
  subap_secondary_diameter_ = subap_diameter_ * sqrt(1 - subap_fill_factor);
  double subap_r = subap_diameter_ / 2;

  // Compute the angle of each of the arms of the Triarm3.
  const double kInitialAngle = -M_PI / 6;
  vector<double> arm_angles;
  for (int i = 0; i < kNumArms; i++) {
    arm_angles.push_back(kInitialAngle + i * 2 * M_PI / kNumArms);
  }

  // Compute the center pixel for each of the subapertures. The s-to-d ratio
  // is a parameter that tells us what the center-to-center distance, s,
  // relative to the subaperture diameter, d.
  int apertures_per_arm = kNumApertures / kNumArms;
  for (int arm = 0; arm < kNumArms; arm++) {
    for (int ap = 0; ap < apertures_per_arm; ap++) {
      double dist_from_center = kTargetRadius - subap_r;
      int x = (int) (kHalfSize + dist_from_center * cos(arm_angles[arm]));
      int y = (int) (kHalfSize + dist_from_center * sin(arm_angles[arm]));
      subaperture_offsets_.push_back(x);
      subaperture_offsets_.push_back(y);
    }
  }
}

Triarm3::~Triarm3() {}

void Triarm3::ExportToZemax(const std::string& aperture_filename,
                            const std::string& obstruction_filename) const {
  double scale = this->aperture_params().encircled_diameter() /
                 this->params().array_size() * 1000;
  double offset = this->params().array_size() * 0.5;

  std::ofstream ap_ofs(aperture_filename.c_str());
  std::ofstream ob_ofs(obstruction_filename.c_str());
  if (!ap_ofs.is_open() || !ob_ofs.is_open()) return;

  for (int ap = 0; ap < kNumApertures; ap++) {
    ap_ofs << "CIR " << (subaperture_offsets_[2*ap] - offset) * scale << " "
                     << (subaperture_offsets_[2*ap+1] - offset) * scale << " "
                     << subap_diameter_ * 0.5 * scale << std::endl;
    ob_ofs << "CIR " << (subaperture_offsets_[2*ap] - offset) * scale << " "
                     << (subaperture_offsets_[2*ap+1] - offset) * scale << " "
                     << subap_secondary_diameter_ * 0.5 * scale << std::endl;
  }
}

Mat Triarm3::GetApertureTemplate() {
  if (mask_.rows > 0) return mask_;

  const int kSize = params().array_size();

  // Allocate the output array.
  Mat output(kSize, kSize, CV_64FC1);
  double* output_data = (double*)output.data;

  double subap_r2 = subap_diameter_ * subap_diameter_ / 4.0;
  double subap_sec_r2 = subap_secondary_diameter_ * subap_secondary_diameter_
                        / 4.0;

  for (int y = 0; y < kSize; y++) {
    for (int x = 0; x < kSize; x++) {
      for (int ap = 0; ap < kNumApertures; ap++) {
        double x_diff = x - subaperture_offsets_[2*ap];
        double y_diff = y - subaperture_offsets_[2*ap + 1];
        double dist2 = x_diff*x_diff + y_diff*y_diff;

        if (dist2 < subap_r2 && dist2 >= subap_sec_r2) {
          output_data[y*kSize + x] += 1;
        }
      }
    }
  }

  double max_val;
  minMaxIdx(output, NULL, &max_val);
  if (max_val > 1) {
    mainLog() << "Error: Triarm3 subapertures are overlapping!" << endl;
  }

  mask_ = output;

  return output;
}

Mat Triarm3::GetOpticalPathLengthDiff() {
  if (opd_.rows > 0) return opd_;

  for (size_t i = 0; i < aberrations().size(); i++) {
    cout << aberrations().at(i) << endl;
  }

  AberrationFactory::ZernikeAberrations(aberrations(),
      params().array_size(), &opd_);

  return opd_;
}

Mat Triarm3::GetOpticalPathLengthDiffEstimate() {
  if (simulation_params().wfe_knowledge() == Simulation::NONE) {
    return Mat(params().array_size(), params().array_size(), CV_64FC1);
  }

  if (opd_.rows == 0) GetOpticalPathLengthDiff();
  if (opd_est_.rows > 0) return opd_est_;

  double knowledge_level = 0;
  switch (simulation_params().wfe_knowledge()) {
    case Simulation::HIGH: knowledge_level = 0.05; break;
    case Simulation::MEDIUM: knowledge_level = 0.1; break;
    default: knowledge_level = 0.2; break;
  }

  mainLog() << "Error in the estimates of piston/tip/tilt: "
            << knowledge_level << " [waves]" << std::endl;

  vector<double>& real_weights = aberrations();
  vector<double> wrong_weights;
  for (size_t i = 0; i < real_weights.size(); i++) {
    wrong_weights.push_back(real_weights[i] +
        (2 * (rand() % 2) - 1) * knowledge_level);
  }

  AberrationFactory::ZernikeAberrations(aberrations(),
      params().array_size(), &opd_est_);

  return opd_est_;
}
