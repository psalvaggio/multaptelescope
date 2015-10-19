// File Description
// Author: Philip Salvaggio

#include "golay6.h"

#include "base/aperture_parameters.pb.h"
#include "base/opencv_utils.h"
#include "base/simulation_config.pb.h"
#include "base/zernike_aberrations.h"
#include "io/logging.h"
#include "optical_designs/cassegrain.h"
#include "optical_designs/compound_aperture_parameters.pb.h"

#include <algorithm>
#include <fstream>
#include <opencv/highgui.h>

using namespace cv;
using namespace std;
using mats::Simulation;
using mats::ApertureParameters;

Golay6::Golay6(const Simulation& params)
    : Aperture(params),
      compound_aperture_() {
  // Copy over the CassegrainRing-specific parameters.
  golay6_params_ = aperture_params().GetExtension(golay6_params);

  // The fill factor of the overall aperture and each of the individual
  // Cassegrain subapertures.
  double fill_factor = aperture_params().fill_factor();
  double subap_fill_factor = golay6_params_.subaperture_fill_factor();

  // Compute the enclosed area and the area of each subapertures [pixels^2]
  double diameter = aperture_params().encircled_diameter();
  double area_enclosed = M_PI * diameter * diameter / 4;
  double area_subap = area_enclosed * fill_factor / 6;

  // Solve for the radius of the subapertures [pixels]
  double subap_diameter = 2 * sqrt(area_subap / (subap_fill_factor * M_PI));

  // Transform the Golay-6 positions.
  vector<double> subaperture_offsets{
    -0.269461, -0.145774,
    -0.136939, -0.375480,
     0.260628, -0.145774,
     0.393150,  0.083930,
    -0.004417,  0.313636,
    -0.269461,  0.313636};

  for (auto& tmp : subaperture_offsets) tmp *= diameter;

  // Make a copy of our SimulationConfig to give to the subapertures.
  mats::Simulation sim;
  sim.CopyFrom(params);  // Rotation smuggled in here

  // Add the Cassegrain array.
  auto cassegrain_array = sim.mutable_aperture_params();
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
    for (int j = 0; j < golay6_params_.aperture_aberrations_size(); j++) {
      if ((i/2) == (size_t)golay6_params_.aperture_aberrations(j).ap_index()) {
        ab_index = j;
        break;
      }
    }

    if (ab_index != -1) {
      const Golay6Parameters::ApertureAberrations& ap_aberrations(
          golay6_params_.aperture_aberrations(ab_index));
      for (int j = 0; j < ap_aberrations.aberration_size(); j++) {
        mats::ZernikeCoefficient* tmp_aberration = cassegrain->add_aberration();
        tmp_aberration->CopyFrom(ap_aberrations.aberration(j));
      }
    }
  }

  // Construct the aperture.
  compound_aperture_.reset(ApertureFactory::Create(sim));
}

Golay6::~Golay6() {}

void Golay6::GetApertureTemplate(Mat_<double>* output) const {
  compound_aperture_->GetApertureMask(output->rows).copyTo(*output);
}

void Golay6::GetOpticalPathLengthDiff(double image_height,
                                      double angle,
                                      Mat_<double>* output) const {
  compound_aperture_->GetWavefrontError(image_height, angle, output);

  Mat_<double> global(output->size());
  ZernikeWavefrontError(image_height, angle, &global);

  *output += global;
}
