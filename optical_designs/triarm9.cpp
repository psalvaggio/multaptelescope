// File Description
// Author: Philip Salvaggio

#include "triarm9.h"

#include "base/aperture_parameters.pb.h"
#include "base/simulation_config.pb.h"
#include "base/zernike_aberrations.h"
#include "io/logging.h"
#include "optical_designs/cassegrain.h"
#include "optical_designs/compound_aperture_parameters.pb.h"
#include "optical_designs/triarm9_parameters.pb.h"

#include <algorithm>
#include <fstream>
#include <opencv/highgui.h>

using namespace cv;
using namespace std;
using mats::Simulation;
using mats::ApertureParameters;

Triarm9::Triarm9(const Simulation& params)
    : Aperture(params), triarm9_params_(), compound_aperture_() {
  // Copy over the Triarm9-specific parameters.
  triarm9_params_ = this->aperture_params().GetExtension(triarm9_params);

  // The fill factor of the overall aperture and each of the individual
  // Cassegrain subapertures.
  double fill_factor = aperture_params().fill_factor();
  double subap_fill_factor = triarm9_params_.subaperture_fill_factor();

  // Compute the enclosed area and the area of each subapertures [pixels^2]
  double diameter = aperture_params().encircled_diameter();
  double area_enclosed = M_PI * diameter * diameter / 4;
  double area_subap = area_enclosed * fill_factor / kNumApertures;

  // Solve for the radius of the subapertures [pixels]
  double subap_r = sqrt(area_subap / (subap_fill_factor * M_PI));
  double subap_diameter = 2 * subap_r;

  // Compute the angle of each of the arms of the Triarm9.
  double initial_angle = -M_PI / 6;
  if (aperture_params().has_rotation()) {
    initial_angle += aperture_params().rotation();
  }
  vector<double> arm_angles;
  for (int i = 0; i < kNumArms; i++) {
    arm_angles.push_back(initial_angle + i * 2 * M_PI / kNumArms);
  }

  // Compute the center pixel for each of the subapertures. The s-to-d ratio
  // is a parameter that tells us what the center-to-center distance, s,
  // relative to the subaperture diameter, d.
  const double kTargetRadius = 0.5 * diameter;
  int apertures_per_arm = kNumApertures / kNumArms;
  double padding = (triarm9_params_.s_to_d_ratio() - 1) * subap_diameter;
  vector<double> subaperture_offsets;
  for (int arm = 0; arm < kNumArms; arm++) {
    for (int ap = 0; ap < apertures_per_arm; ap++) {
      double dist_from_center = (kTargetRadius - (2*ap + 1) * subap_r) -
                                ap * padding;
      double x = dist_from_center * cos(arm_angles[arm]);
      double y = dist_from_center * sin(arm_angles[arm]);
      subaperture_offsets.push_back(x);
      subaperture_offsets.push_back(y);
    }
  }
  
  // Make a copy of our SimulationConfig to give to the subapertures.
  Simulation sim;
  sim.CopyFrom(params);  // Rotation smuggled in here

  // Add the Cassegrain array.
  auto cassegrain_array = sim.mutable_aperture_params();
  cassegrain_array->clear_rotation();
  cassegrain_array->set_type(ApertureParameters::COMPOUND);

  auto cassegrain_array_ext =
      cassegrain_array->MutableExtension(compound_aperture_params);
  cassegrain_array_ext->set_combine_operation(CompoundApertureParameters::OR);
  for (size_t i = 0; i < subaperture_offsets.size(); i += 2) {
    auto cassegrain = cassegrain_array_ext->add_aperture();
    cassegrain->set_type(ApertureParameters::CASSEGRAIN);
    cassegrain->set_encircled_diameter(subap_diameter);
    cassegrain->set_fill_factor(subap_fill_factor);
    cassegrain->set_offset_x(subaperture_offsets[i]);
    cassegrain->set_offset_y(subaperture_offsets[i+1]);

    int ab_index = -1;
    for (int j = 0; j < triarm9_params_.aperture_aberrations_size(); j++) {
      if ((i/2) == (size_t)triarm9_params_.aperture_aberrations(j).ap_index()) {
        ab_index = j;
        break;
      }
    }

    if (ab_index != -1) {
      const Triarm9Parameters::ApertureAberrations& ap_aberrations(
          triarm9_params_.aperture_aberrations(ab_index));
      for (int j = 0; j < ap_aberrations.aberration_size(); j++) {
        mats::ZernikeCoefficient* tmp_aberration = cassegrain->add_aberration();
        tmp_aberration->CopyFrom(ap_aberrations.aberration(j));
      }
    }
  }

  // Construct the aperture.
  compound_aperture_.reset(ApertureFactory::Create(sim));
}

Triarm9::~Triarm9() {}

void Triarm9::GetApertureTemplate(Mat_<double>* output) const {
  compound_aperture_->GetApertureMask(output->rows).copyTo(*output);
}

void Triarm9::GetOpticalPathLengthDiff(double image_height,
                                       double angle,
                                       Mat_<double>* output) const {
  compound_aperture_->GetWavefrontError(image_height, angle, output);

  Mat_<double> global(output->size());
  ZernikeWavefrontError(image_height, angle, &global);

  *output += global;
}
