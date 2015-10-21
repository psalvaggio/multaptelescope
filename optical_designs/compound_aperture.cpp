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

using mats::Simulation;
using namespace std;
using namespace cv;

CompoundAperture::CompoundAperture(const Simulation& params)
    : Aperture(params),
      compound_params_(
          aperture_params().GetExtension(compound_aperture_params)),
      apertures_(),
      sim_configs_() {
  // Construct each subaperture.
  for (int i = 0; i < compound_params_.aperture_size(); i++) {
    sim_configs_.emplace_back();

    auto& tmp_conf(sim_configs_.back());
    tmp_conf.CopyFrom(params);
    auto ap_params = tmp_conf.mutable_aperture_params();
    ap_params->CopyFrom(compound_params_.aperture(i));

    apertures_.emplace_back(ApertureFactory::Create(tmp_conf));
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

    max_diameter = max(max_diameter, 2*tmp_radius);
  }
  return max_diameter;
}

void CompoundAperture::GetApertureTemplate(Mat_<double>* output) const {
  Mat& mask = *output;
  const int kSize = mask.rows;

  // Determine how we will be combining the masks
  int combine_op = compound_params_.combine_operation();

  // Create the mask for each sub-aperture
  vector<Mat_<double>> masks;
  GenerateSubapertureHelper(kSize, &masks,
      [] (const Aperture* subap, Mat* mask) {
        subap->GetApertureMask(mask->rows).copyTo(*mask);
      });

  // Combine the sub-apertures
  if (combine_op == CompoundApertureParameters::AND ||
      combine_op == CompoundApertureParameters::AND_WFE_ADD) {
    mask = Scalar(1);
    for (const auto& subap_mask : masks) {
      multiply(mask, subap_mask, mask);
    }
  } else if (combine_op == CompoundApertureParameters::OR) {
    mask = Scalar(0);
    for (const auto& subap_mask : masks) {
      cv::max(mask, subap_mask, mask);
    }
  }

  RotateArray(output);
}

void CompoundAperture::GetOpticalPathLengthDiff(double image_height,
                                                double angle,
                                                Mat_<double>* output) const {
  Mat_<double>& opd = *output;
  const int kSize = opd.rows;

  // Determine how we will be combining the masks
  int combine_op = compound_params_.combine_operation();

  vector<Mat_<double>> wfes;
  GenerateSubapertureHelper(kSize, &wfes,
      [image_height, angle] (const Aperture* subap, Mat_<double>* opd) {
        subap->GetWavefrontError(image_height, angle, opd);
      });

  if (combine_op == CompoundApertureParameters::AND) {
    opd = wfes[compound_params_.wfe_index()];
  } else if (combine_op == CompoundApertureParameters::AND_WFE_ADD ||
             combine_op == CompoundApertureParameters::OR) {
    opd = 0;
    for (const auto& tmp_wfe : wfes) opd += tmp_wfe;
  }

  RotateArray(output);
}

void CompoundAperture::GenerateSubapertureHelper(
    int array_size,
    vector<Mat_<double>>* subaps,
    function<void(const Aperture*, Mat_<double>*)> subap_generator) const {
  int array_half_size = array_size / 2;

  // Get the scale of the mask. [m / pixel]
  const double kMaskScale = encircled_diameter() / array_size;

  // Create the mask for each sub-aperture
  for (const auto& subap : apertures_) {
    int subap_size = round(subap->encircled_diameter() / kMaskScale);
    int subap_half_size = subap_size / 2;

    double subap_center_x = subap->aperture_params().offset_x() / kMaskScale;
    double subap_center_y = subap->aperture_params().offset_y() / kMaskScale;

    int subap_x0 = round(subap_center_x + array_half_size - subap_half_size);
    int subap_y0 = round(subap_center_y + array_half_size - subap_half_size);
    int subap_x1 = subap_x0 + subap_size;
    int subap_y1 = subap_y0 + subap_size;

    if (subap_x0 < 0 || subap_y0 < 0 ||
        subap_x1 > array_size || subap_y1 > array_size) {
      mainLog() << "Error: CompoundAperture: One of the subapertures exceeds "
                << "the bounds of the encircled diameter." << endl;
      return;
    }

    subaps->emplace_back(array_size, array_size);
    Mat_<double>& subap_array = subaps->back();

    subap_array = 0;

    Mat_<double> subap_region = subap_array(Range(subap_y0, subap_y1),
                                            Range(subap_x0, subap_x1));
    subap_generator(subap.get(), &subap_region);
  }
}

void CompoundAperture::RotateArray(Mat_<double>* array) const {
  if (aperture_params().has_rotation() && aperture_params().rotation() != 0) {
    Mat rotation = getRotationMatrix2D(
        Point2f(array->cols / 2, array->rows / 2),
        aperture_params().rotation(),
        1);
    warpAffine(*array, *array, rotation, array->size());
  }
}
