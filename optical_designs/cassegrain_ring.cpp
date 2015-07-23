// File Description
// Author: Philip Salvaggio

#include "cassegrain_ring.h"

#include "base/aperture_parameters.pb.h"
#include "base/opencv_utils.h"
#include "base/simulation_config.pb.h"
#include "io/logging.h"
#include "optical_designs/cassegrain.h"
#include "optical_designs/cassegrain_ring_parameters.pb.h"
#include "optical_designs/compound_aperture_parameters.pb.h"

#include <algorithm>
#include <fstream>
#include <opencv/highgui.h>

using namespace cv;
using namespace std;
using mats::Simulation;
using mats::ApertureParameters;

CassegrainRing::CassegrainRing(const mats::SimulationConfig& params,
                               int sim_index)
    : Aperture(params, sim_index),
      compound_aperture_() {
  // Copy over the CassegrainRing-specific parameters.
  ring_params_ = this->aperture_params().GetExtension(cassegrain_ring_params);

  // Aperture construction parameters.
  const int kNumApertures = ring_params_.num_apertures();
  const double kInitialAngle = ring_params_.angle_offset();

  // The fill factor of the overall aperture and each of the individual
  // Cassegrain subapertures.
  double fill_factor = aperture_params().fill_factor();
  double subap_fill_factor = ring_params_.subaperture_fill_factor();

  // Compute the enclosed area and the area of each subapertures [pixels^2]
  double diameter = aperture_params().encircled_diameter();
  double radius = diameter / 2;
  double area_enclosed = M_PI * diameter * diameter / 4;
  double area_subap = area_enclosed * fill_factor / kNumApertures;

  // Solve for the radius of the subapertures [pixels]
  double subap_diameter = 2 * sqrt(area_subap / (subap_fill_factor * M_PI));
  double subap_r = subap_diameter / 2;

  // Compute the center pixel for each of the subapertures.
  vector<double> subaperture_offsets;
  for (int i = 0; i < kNumApertures; i++) {
    double angle = kInitialAngle + i * 2 * M_PI / kNumApertures;

    double dist_from_center = radius - subap_r;
    double x = dist_from_center * cos(angle);
    double y = dist_from_center * sin(angle);
    subaperture_offsets.push_back(x);
    subaperture_offsets.push_back(y);
  }

  // Make a copy of our SimulationConfig to give to the subapertures.
  mats::SimulationConfig conf;
  conf.CopyFrom(params);
  conf.clear_simulation();
  mats::Simulation* sim = conf.add_simulation();
  sim->CopyFrom(params.simulation(sim_index));

  // Add the top-level compound aperture. This is the AND of the cassegrain
  // subapertures (compound) and a circular aperture containing the shared
  // wavefront error.
  ApertureParameters* compound_params = sim->mutable_aperture_params();
  compound_params->set_type(ApertureParameters::COMPOUND);
  CompoundApertureParameters* compound_ext = 
      compound_params->MutableExtension(compound_aperture_params);
  compound_ext->set_combine_operation(CompoundApertureParameters::AND_WFE_ADD);
  compound_ext->set_wfe_index(0);

  // Add the circular aperture with the wavefront error.
  ApertureParameters* circular_mask = compound_ext->add_aperture();
  circular_mask->set_type(ApertureParameters::CIRCULAR);
  circular_mask->set_encircled_diameter(encircled_diameter());
  for (int i = 0; i < aperture_params().aberration_size(); i++) {
    mats::ZernikeCoefficient* tmp_ab = circular_mask->add_aberration();
    tmp_ab->CopyFrom(aperture_params().aberration(i));
  }

  // Add the Cassegrain array.
  ApertureParameters* cassegrain_array = compound_ext->add_aperture();
  cassegrain_array->set_type(ApertureParameters::COMPOUND);
  CompoundApertureParameters* cassegrain_array_ext =
      cassegrain_array->MutableExtension(compound_aperture_params);
  cassegrain_array_ext->set_combine_operation(CompoundApertureParameters::OR);
  for (size_t i = 0; i < subaperture_offsets.size(); i += 2) {
    ApertureParameters* cassegrain = cassegrain_array_ext->add_aperture();
    cassegrain->set_type(ApertureParameters::CASSEGRAIN);
    cassegrain->set_encircled_diameter(subap_diameter);
    cassegrain->set_fill_factor(subap_fill_factor);
    cassegrain->set_offset_x(subaperture_offsets[i]);
    cassegrain->set_offset_y(subaperture_offsets[i+1]);

    int ab_index = -1;
    for (int j = 0; j < ring_params_.aperture_aberrations_size(); j++) {
      if ((i/2) == (size_t)ring_params_.aperture_aberrations(j).ap_index()) {
        ab_index = j;
        break;
      }
    }

    if (ab_index != -1) {
      const CassegrainRingParameters::ApertureAberrations& ap_aberrations(
          ring_params_.aperture_aberrations(ab_index));
      for (int j = 0; j < ap_aberrations.aberration_size(); j++) {
        mats::ZernikeCoefficient* tmp_aberration = cassegrain->add_aberration();
        tmp_aberration->CopyFrom(ap_aberrations.aberration(j));
      }
    }
  }

  // Construct the aperture.
  compound_aperture_.reset(ApertureFactory::Create(conf, 0));
}

CassegrainRing::~CassegrainRing() {}

void CassegrainRing::GetApertureTemplate(Mat_<double>* output) const {
  compound_aperture_->GetApertureMask(output->rows).copyTo(*output);
}

void CassegrainRing::GetOpticalPathLengthDiff(Mat_<double>* output) const {
  compound_aperture_->GetWavefrontError(output->rows).copyTo(*output);
}

void CassegrainRing::GetOpticalPathLengthDiffEstimate(
    Mat_<double>* output) const {
  compound_aperture_->GetWavefrontErrorEstimate(output->rows).copyTo(*output);
}
