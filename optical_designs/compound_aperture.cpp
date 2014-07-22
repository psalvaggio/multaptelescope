// File Description
// Author: Philip Salvaggio

#include "compound_aperture.h"

CompoundAperture::CompoundAperture(const mats::SimulationConfig& params,
                                   int sim_index)
    : Aperture(params, sim_index),
      compound_params_(
          this->aperture_params().GetExtension(compound_aperture_params)),
      mask_(),
      opd_(),
      opd_est_() {}

CompoundAperture::~CompoundAperture() {}

cv::Mat CompoundAperture::GetApertureTemplate() {
}

cv::Mat CompoundAperture::GetOpticalPathLengthDiff() {
}

cv::Mat CompoundAperture::GetOpticalPathLengthDiffEstimate() {
}
