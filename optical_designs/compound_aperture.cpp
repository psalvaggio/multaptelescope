// File Description
// Author: Philip Salvaggio

#include "compound_aperture.h"

#include "base/aperture_parameters.pb.h"
#include "base/assertions.h"
#include "base/opencv_utils.h"
#include "base/str_utils.h"
#include "base/simulation_config.pb.h"
#include "io/logging.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

CompoundAperture::CompoundAperture(const mats::SimulationConfig& params,
                                   int sim_index)
    : Aperture(params, sim_index),
      compound_params_(
          this->aperture_params().GetExtension(compound_aperture_params)),
      apertures_(),
      sim_configs_(),
      opd_(),
      opd_est_() {

  // To construct the subapertures, we need a SimulationConfig that makes it
  // look like that is the top-level aperture (not ideal, I know).
  // Make a copy of our SimulationConfig and Simulation.
  mats::SimulationConfig conf;
  conf.CopyFrom(params);
  conf.clear_simulation();
  mats::Simulation* sim = conf.add_simulation();
  sim->CopyFrom(params.simulation(sim_index));

  // Construct each subaperture.
  for (int i = 0; i < compound_params_.aperture_size(); i++) {
    sim_configs_.push_back(mats::SimulationConfig());

    mats::SimulationConfig& tmp_conf(sim_configs_.back());
    tmp_conf.CopyFrom(conf);
    mats::ApertureParameters* ap_params =
        tmp_conf.mutable_simulation(0)->mutable_aperture_params();
    ap_params->CopyFrom(compound_params_.aperture(i));

    apertures_.push_back(std::move(std::unique_ptr<Aperture>(
        ApertureFactory::Create(tmp_conf, 0))));
  }
}

CompoundAperture::~CompoundAperture() {}

double CompoundAperture::GetEncircledDiameter() const {
  if (aperture_params().has_encircled_diameter()) {
    return aperture_params().encircled_diameter();
  }
  
  double max_diameter = 0;
  for (size_t i = 0; i < apertures_.size(); i++) {
    double x = apertures_[i]->aperture_params().offset_x();
    double y = apertures_[i]->aperture_params().offset_y();
    double r = apertures_[i]->encircled_diameter() / 2.0;
    double tmp_radius = sqrt(x*x + y*y) + r;

    max_diameter = std::max(max_diameter, 2*tmp_radius);
  }
  return max_diameter;
}

cv::Mat CompoundAperture::GetApertureTemplate() const {
  // Determine how we will be combining the masks
  int combine_op = compound_params_.combine_operation();

  // Get the size of the arrays.
  const int kSize = params().array_size();

  std::vector<cv::Mat> masks;
  std::vector<cv::Mat> wfes;
  std::vector<cv::Mat> wfe_ests;
  for (size_t i = 0; i < apertures_.size(); i++) {
    // Get the subaperture mask.
    std::vector<cv::Mat> ap_data;
    ap_data.push_back(apertures_[i]->GetApertureMask());
    
    // Determine if we need the WFE and get it if so.
    bool dont_need_wfe = combine_op == CompoundApertureParameters::AND &&
        i != (size_t)compound_params_.wfe_index();
    if (!dont_need_wfe) {
      ap_data.push_back(apertures_[i]->GetWavefrontError());
      ap_data.push_back(apertures_[i]->GetWavefrontErrorEstimate());
    }

    // Scale the subaperture based on its diameter
    double scale = apertures_[i]->encircled_diameter() / encircled_diameter();
    int new_size = round(ap_data[0].rows * scale);
    std::vector<cv::Mat> scaled;
    for (size_t j = 0; j < ap_data.size(); j++) {
      scaled.push_back(cv::Mat());
      cv::resize(ap_data[j], scaled.back(), cv::Size(new_size, new_size),
          0, 0, cv::INTER_NEAREST);
    }

    // We need all of the apertures to be the same size. This means that we
    // need to add zero padding to the periphery of the subapertures.
    if (scaled[0].rows < kSize) { 
      for (size_t j = 0; j < scaled.size(); j++) {
        cv::Mat tmp_scaled = cv::Mat::zeros(kSize, kSize, CV_64FC1);
        double* large_data = reinterpret_cast<double*>(tmp_scaled.data);
        double* small_data = reinterpret_cast<double*>(scaled[j].data);

        int pixels_to_pad = kSize - new_size;
        int pad = pixels_to_pad / 2;

        for (int y = 0; y < scaled[j].rows; y++) {
          for (int x = 0; x < scaled[j].cols; x++) {
            large_data[(pad + y)*kSize + (pad + x)] =
                small_data[y*scaled[j].rows + x];
          }
        }

        scaled[j] = tmp_scaled;
      }
    } else if (scaled[0].rows > kSize) {
      for (size_t j = 0; j < scaled.size(); j++) {
        cv::Mat tmp_scaled = cv::Mat::zeros(kSize, kSize, CV_64FC1);
        double* large_data = reinterpret_cast<double*>(scaled[j].data);
        double* small_data = reinterpret_cast<double*>(tmp_scaled.data);

        int pixels_to_trim = new_size - kSize;
        int pad = pixels_to_trim / 2;

        int large_rows = scaled[j].rows;

        for (int y = 0; y < kSize; y++) {
          for (int x = 0; x < kSize; x++) {
            small_data[y*kSize  + x] =
                large_data[(pad + y)*large_rows + (pad + x)];
          }
        }

        scaled[j] = tmp_scaled;
      }
    }

    double pixel_scale = kSize / encircled_diameter();
    int offset_x = round(apertures_[i]->aperture_params().offset_x() *
        pixel_scale);
    int offset_y = round(apertures_[i]->aperture_params().offset_y() *
        pixel_scale);

    std::vector<cv::Mat>* output_vecs[] = {&masks, &wfes, &wfe_ests};
    if (offset_x != 0 || offset_y != 0) {
      cv::Point2f shift(offset_x, offset_y);
      for (size_t j = 0; j < scaled.size(); j++) {
        output_vecs[j]->push_back(cv::Mat());
        circshift(scaled[j], output_vecs[j]->back(), shift, cv::BORDER_CONSTANT,
            cv::Scalar(0));
      }
    } else {
      for (size_t j = 0; j < scaled.size(); j++) {
        output_vecs[j]->push_back(scaled[j]);
      }
    }
  }

  cv::Mat result;
  if (combine_op == CompoundApertureParameters::AND) {
    result = cv::Mat::ones(kSize, kSize, CV_64F);
    for (size_t i = 0; i < masks.size(); i++) {
      cv::multiply(result, masks[i], result);
    }
    opd_ = wfes[0];
    opd_est_ = wfe_ests[0];
  } else if (combine_op == CompoundApertureParameters::AND_WFE_ADD) {
    result = cv::Mat::ones(kSize, kSize, CV_64F);
    for (size_t i = 0; i < masks.size(); i++) {
      cv::multiply(result, masks[i], result);
    }

    opd_ = cv::Mat::zeros(kSize, kSize, CV_64FC1);
    opd_est_ = cv::Mat::zeros(kSize, kSize, CV_64FC1);
    for (size_t i = 0; i < masks.size(); i++) {
      opd_ += wfes[i];
      opd_est_ += wfe_ests[i];
    }
  } else if (combine_op == CompoundApertureParameters::OR) {
    result = cv::Mat::zeros(kSize, kSize, CV_64F);
    for (size_t i = 0; i < masks.size(); i++) {
      cv::max(result, masks[i], result);
    }

    opd_ = cv::Mat::zeros(kSize, kSize, CV_64FC1);
    opd_est_ = cv::Mat::zeros(kSize, kSize, CV_64FC1);
    for (size_t i = 0; i < masks.size(); i++) {
      opd_ += wfes[i];
      opd_est_ += wfe_ests[i];
    }
  }

  return result;
}

cv::Mat CompoundAperture::GetOpticalPathLengthDiff() const {
  GetApertureMask();  // Will recompute opd_ is needed.
  return opd_;
}

cv::Mat CompoundAperture::GetOpticalPathLengthDiffEstimate() const {
  GetApertureMask();  // Will recompute opd_est_ if needed.
  return opd_est_;
}
