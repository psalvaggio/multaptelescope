// File Description
// Author: Philip Salvaggio

#include "telescope.h"

#include "base/pupil_function.h"
#include "base/opencv_utils.h"
#include "optical_designs/aperture.h"

#include <algorithm>

using std::vector;
using namespace cv;

namespace mats {

Telescope::Telescope(const SimulationConfig& sim_config,
                     int sim_index,
                     const ApertureParameters& ap_params,
                     const DetectorParameters& det_params)
    : aperture_(ApertureFactory::Create(sim_config, sim_index, ap_params)),
      detector_(new Detector(det_params, sim_config, sim_index)) {}

// In this model, the user specifies the pixel pitch of the detector, the
// altitude at which the telescope is flying, and the pixel size on the ground.
// Thus, the focal length must be a calculated parameter.
double Telescope::FocalLength() const {
  return detector_->sim_params().altitude() *
         detector_->det_params().pixel_pitch() /
         detector_->simulation().gsd();  // [m]
}

void Telescope::Image(const std::vector<Mat>& radiance,
                      const std::vector<double>& wavelength,
                      std::vector<Mat>* image) {
}

void Telescope::ComputeOtf(const vector<double>& wavelengths,
                           std::vector<Mat>* otf) {
  ComputeApertureOtf(wavelengths, otf);
}

void Telescope::ComputeApertureOtf(const vector<double>& wavelengths,
                                   vector<Mat>* otf) {
  // Array sizes
  const int kOtfSize = aperture_->params().array_size();
  const int kNumRows = detector_->rows();
  const int kNumCols = detector_->cols();

  Mat aperture_wfe = aperture_->GetWavefrontError();

  // The OTF varies drastically with respect to wavelength. So, we will be
  // calculating an OTF for each spectral band in our input radiance data.
  for (size_t i = 0; i < wavelengths.size(); i++) {
    // Get the aberrated pupil function from the aperture.
    PupilFunction pupil_func;
    aperture_->GetPupilFunction(aperture_wfe, wavelengths[i], &pupil_func);

    double aperture_scale = pupil_func.meters_per_pixel();

    // The coherent OTF is given by p[lamda * f * xi, lamda * f * eta],
    // where xi and eta are in [cyc/m]. The pixel pitch factor converts from
    // [cyc/pixel], which we want for degrading the image, and [cyc/m], which
    // is what is used by the pupil function.
    double pupil_scale = wavelengths[i] * FocalLength() /
                         detector_->pixel_pitch();

    // Determine the region of the pupil function OTF that we need. This may
    // be smaller or larger than the original array. If it is smaller, we are
    // getting rid of the padding around the non-zero OTF, meaning the OTF
    // will be wider and we will have better resolution. This occurs are
    // wavelengths shorter than the reference. If the region is larger, we
    // are adding zero padding, the OTF will be smaller and resolution will
    // suffer. This occurs at wavelengths larger than the reference.
    Range otf_range;
    otf_range.start = kOtfSize / 2 -
                      int(0.5 * pupil_scale / aperture_scale);
    otf_range.end = kOtfSize / 2 +
                    int(0.5 * pupil_scale / aperture_scale) + 1;

    // We don't want to deal with the wrapping that occurs with FFT's, so we
    // will shift zero frequency to the middle of the array, so cropping can
    // be done in a single operation.
    Mat unscaled_otf = FFTShift(pupil_func.OpticalTransferFunction());
    Mat scaled_otf;

    // If the range is larger than the image, we need to add zero-padding.
    // Essentially, we will be downsampling the OTF and copying it into the
    // middle of a new array. If the range is smaller, it's a simple upscaling
    // to the required resolution.
    if (otf_range.start < 0 || otf_range.end > kOtfSize) {
      double scaling_size = otf_range.end - otf_range.start;
      int new_width = int(kOtfSize*kOtfSize / scaling_size);
      otf_range.start = kOtfSize / 2 - (new_width + 1) / 2;
      otf_range.end = kOtfSize / 2 + new_width / 2;

      scaled_otf = Mat::zeros(kOtfSize, kOtfSize, CV_64FC2);

      vector<Mat> unscaled_planes, scaled_planes;
      split(unscaled_otf, unscaled_planes);
      for (int i = 0; i < 2; i++) {
        Mat tmp_scaled;
        resize(unscaled_planes[i], tmp_scaled, Size(new_width, new_width),
               0, 0, INTER_NEAREST);

        scaled_planes.push_back(Mat::zeros(kOtfSize, kOtfSize, CV_64FC1));
        tmp_scaled.copyTo(scaled_planes[i](otf_range, otf_range));
      }

      merge(scaled_planes, scaled_otf);
    } else {
      resize(unscaled_otf(otf_range, otf_range), scaled_otf,
             Size(kNumRows, kNumCols), 0, 0, INTER_NEAREST);
    }
    otf->push_back(FFTShift(scaled_otf));
  }
}

}