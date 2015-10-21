// File Description
// Author: Philip Salvaggio

#include "telescope.h"

#include "base/detector.h"
#include "base/opencv_utils.h"
#include "base/pupil_function.h"
#include "base/str_utils.h"
#include "base/system_otf.h"
#include "io/logging.h"
#include "optical_designs/aperture.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <algorithm>
#include <numeric>

using namespace std;
using namespace cv;

namespace mats {

Telescope::Telescope(const SimulationConfig& sim_config,
                     int sim_index,
                     const DetectorParameters& det_params)
    : sim_config_(sim_config),
      aperture_(ApertureFactory::Create(sim_config.simulation(sim_index))),
      detector_(new Detector(det_params)),
      include_detector_footprint_(false),
      parallelism_(false) {}

Telescope::~Telescope() {}

const Simulation& Telescope::simulation() const {
  return aperture_->simulation_params();
}

// In this model, the user specifies the pixel pitch of the detector, the
// altitude at which the telescope is flying, and the pixel size on the ground.
// Thus, the focal length must be a calculated parameter.
double Telescope::FocalLength() const {
  const auto& sim = simulation();
  if (sim.has_focal_length()) return sim.focal_length();

  return sim_config_.altitude() * detector_->pixel_pitch() / sim.gsd();  // [m]
}

double Telescope::FNumber() const {
  return FocalLength() / aperture_->encircled_diameter();
}

double Telescope::GNumber(double lambda) const {
  double f_number = FNumber();

  vector<double> transmission;
  vector<double> wavelengths{lambda};
  GetTransmissionSpectrum(wavelengths, &transmission);

  return (1 + 4 * f_number * f_number) /
         (M_PI * transmission[0] * aperture_->fill_factor());
}

double Telescope::GetEffectiveQ(
    const vector<double>& wavelengths,
    const vector<double>& spectral_weighting) const {
  double fnumber = FNumber();
  double p = detector_->pixel_pitch();
  double q = 0;
  double total_weight = 0;
  for (size_t i = 0; i < wavelengths.size(); i++) {
    double weight = spectral_weighting[max(i, spectral_weighting.size() - 1)];
    q += wavelengths[i] * fnumber * weight / p;
    total_weight += weight;
  }
  return q / total_weight;
}

void Telescope::Image(const vector<Mat>& radiance,
                      const vector<double>& wavelength,
                      vector<Mat>* image,
                      vector<Mat>* otf) {
  vector<Mat> blurred_irradiance(radiance.size());

  // If we have off-axis aberration, we need to compute the output image in
  // various isoplanatic regions and then reconstruct the image
  if (aperture_->HasOffAxisAberration()) {
    const int kRadialZones = simulation().radial_zones();
    const int kAngularZones = simulation().angular_zones();
    const double kRadialZoneWidth = 1 / max(1., kRadialZones - 1.);
    const double kAngularZoneWidth = 2 * M_PI / kAngularZones;

    // Allocate the blurred irradiance, we'll be building up the result one
    // isoplanatic region at a time
    for (size_t i = 0; i < radiance.size(); i++) {
      blurred_irradiance[i] = Mat::zeros(radiance[i].size(), CV_64F);
    }

    // Loop through each isoplanatic region
    Mat_<double> isoplanatic_region(radiance[0].size());
    for (int i = 0; i < kRadialZones; i++) {
      for (int j = 0; j < kAngularZones; j++) {
        cout << "Processing zone r " << (i+1) << "/" << kRadialZones
             << ", theta " << (j+1) << "/" << kAngularZones << endl;
        // Compute the interpolation weights for this region
        IsoplanaticRegion(i, j, &isoplanatic_region);

        // Compute the spactral OTF in the current isoplanatic region
        vector<Mat> spectral_otfs;
        ComputeOtf(wavelength, i * kRadialZoneWidth, j * kAngularZoneWidth,
                   &spectral_otfs);

        // Perform image degradation and add to the total blurred image
        Mat degraded;
        for (size_t k = 0; k < radiance.size(); k++) {
          DegradeImage(radiance[k], spectral_otfs[k], &degraded);
          multiply(degraded, isoplanatic_region, degraded,
              1 / GNumber(wavelength[k]));
          blurred_irradiance[k] += degraded;
        }

        if (i == 0) j = kAngularZones;
      }
    }
  } else {
    // Compute the OTF for each of the spectral points in the input data.
    vector<Mat> spectral_otfs;
    ComputeOtf(wavelength, 0, 0, &spectral_otfs);

    for (size_t i = 0; i < radiance.size(); i++) {
      DegradeImage(radiance[i], spectral_otfs[i], &(blurred_irradiance[i]));
      blurred_irradiance[i] /= GNumber(wavelength[i]);
    }
  }
                     
  // Compute the detector response
  vector<Mat> electrons;
  double int_time = simulation().integration_time();
  detector_->ResponseElectrons(blurred_irradiance, wavelength, int_time,
                               &electrons);
  detector_->Quantize(electrons, image);
}

void Telescope::DegradeImage(const Mat& radiance,
                             const Mat& spectral_otf,
                             Mat* degraded) const {
  const int kRows = detector_->rows();
  const int kCols = detector_->cols();

  Mat radiance_roi;
  GetImagingRegion(radiance, &radiance_roi);
  if (radiance_roi.rows == 0) {
    *degraded = Mat::zeros(detector_->rows(), detector_->cols(), CV_64FC1);
    return;
  }

  Mat img_fft, blurred_fft;
  dft(radiance_roi, img_fft, DFT_COMPLEX_OUTPUT);
  OtfDegrade(img_fft, spectral_otf, &blurred_fft);

  dft(blurred_fft, *degraded, DFT_INVERSE | DFT_SCALE | DFT_REAL_OUTPUT);

  if (degraded->cols != kCols) {
    resize(*degraded, *degraded, Size(kCols, kRows), 0, 0, INTER_AREA);
  }
}

void Telescope::OtfDegrade(const Mat& radiance_fft,
                           const Mat& spectral_otf,
                           Mat* degraded) const {
  Mat otf(radiance_fft.size(), CV_64FC2);
  otf.setTo(Scalar(0, 0));
  int row_start = radiance_fft.rows / 2 - detector_->rows() / 2;
  int col_start = radiance_fft.cols / 2 - detector_->cols() / 2;
  Range row_range(row_start, row_start + detector_->rows()),
        col_range(col_start, col_start + detector_->cols());

  resize(FFTShift(spectral_otf), otf(row_range, col_range),
         Size(detector_->cols(), detector_->rows()));

  mulSpectrums(radiance_fft, IFFTShift(otf), *degraded, 0);
}

void Telescope::GetImagingRegion(const Mat& radiance, Mat* roi) const {
  const int kRows = detector_->rows();
  const int kCols = detector_->cols();
  const double kAspectRatio = double(kRows) / kCols;

  if (kRows == 0 || kCols == 0) {
    mainLog() << "Error: Detector has zero size." << endl;
    return;
  }

  if (radiance.rows < kRows || radiance.cols < kCols) {
    mainLog() << "Error: given radiance image should be at least as large "
              << "as the detector." << endl;
    return;
  }

  int rows, cols;
  if (kRows > kCols) {
    rows = radiance.rows;
    cols = round(rows / kAspectRatio);
  } else {
    cols = radiance.cols;
    rows = round(cols * kAspectRatio);
  }

  if (cols > radiance.cols || rows > radiance.rows) {
    mainLog() << "Error: could not extract an imaging region from the given"
              << "image." << endl;
    return;
  }

  *roi = radiance(Range(0, rows), Range(0, cols));
}


void Telescope::ComputeOtf(const vector<double>& wavelengths,
                           double image_height,
                           double angle,
                           vector<Mat>* otf) const {
  const int kOtfSize = sim_config_.array_size();

  vector<Mat_<complex<double>>> ap_otf;
  ComputeApertureOtf(wavelengths, image_height, angle, &ap_otf);

  double int_time = simulation().integration_time();

  SystemOtf wave_invar_sys_otf;
  wave_invar_sys_otf.PushOtf(detector_->GetSmearOtf(0, 0, int_time,
                                                    kOtfSize, kOtfSize));
  wave_invar_sys_otf.PushOtf(detector_->GetJitterOtf(0, int_time,
                                                     kOtfSize, kOtfSize));
  if (include_detector_footprint_) {
    wave_invar_sys_otf.PushOtf(detector_->GetSamplingOtf(kOtfSize, kOtfSize));
  }
  Mat_<complex<double>> wave_invariant_otf = wave_invar_sys_otf.GetOtf();

  for (size_t i = 0; i < wavelengths.size(); i++) {
    otf->emplace_back();
    mulSpectrums(ap_otf[i], wave_invariant_otf, otf->back(), 0);
  }
}

void Telescope::ComputeEffectiveOtf(const vector<double>& wavelengths,
                                    const vector<double>& weights,
                                    double image_height,
                                    double angle,
                                    cv::Mat* otf) const {
  if (!otf || wavelengths.size() == 0 || weights.size() < wavelengths.size()) {
    cerr << "Telescope::ComputeEffectiveOtf: Invalid Input" << endl;
    return;
  }

  vector<Mat> spectral_otf;
  ComputeOtf(wavelengths, image_height, angle, &spectral_otf);

  double total_weight = accumulate(begin(weights), end(weights), 0.0);

  spectral_otf[0].copyTo(*otf);
  (*otf) *= weights[0];
  for (size_t i = 1; i < spectral_otf.size(); i++) {
    (*otf) += (weights[i] / total_weight) * spectral_otf[i];
  }
}

void Telescope::GetTransmissionSpectrum(
    const vector<double>& wavelengths,
    vector<double>* transmission) const {
  if (!transmission) return;

  // We're just going to use a flat transmittance for the optics.
  const double kTransmittance = 0.9;
  transmission->clear();
  transmission->resize(wavelengths.size(), kTransmittance);
}

void Telescope::ComputeApertureOtf(const vector<double>& wavelengths,
                                   double image_height,
                                   double angle,
                                   vector<Mat_<complex<double>>>* otf) const {
  otf->clear();

  // Array sizes
  const int kOtfSize = sim_config_.array_size();

  // Get the aberrated pupil function from the aperture.
  vector<PupilFunction> pupil_funcs;
  aperture_->GetPupilFunction(wavelengths, image_height, angle,
                              kOtfSize, sim_config_.reference_wavelength(),
                              &pupil_funcs);

  for (size_t i = 0; i < wavelengths.size(); i++) {
    otf->emplace_back();

    // The coherent OTF is given by p[lamda * f * xi, lamda * f * eta],
    // where xi and eta are in [cyc/m]. The pixel pitch factor converts from
    // [cyc/pixel], which we want for degrading the image, and [cyc/m], which
    // is what is used by the pupil function.
    double pupil_scale = wavelengths[i] * FocalLength() /
                         detector_->pixel_pitch();
    
    // Get the scale of the aperture, how many meters does each pixel represent
    // on the aperture plane [m/pix]
    double aperture_scale = pupil_funcs[i].meters_per_pixel();

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
    Mat unscaled_otf = FFTShift(pupil_funcs[i].OpticalTransferFunction());
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
             Size(kOtfSize, kOtfSize), 0, 0, INTER_NEAREST);
    }

    Mat& spectral_otf = otf->back();
    spectral_otf = IFFTShift(scaled_otf);

    spectral_otf.at<complex<double>>(0, 0) /=
        std::abs(spectral_otf.at<complex<double>>(0, 0));
  }
}


void Telescope::IsoplanaticRegion(int radial_idx,
                                  int angular_idx,
                                  Mat_<double>* isoplanatic_region) const {
  const int kNumRadialZones = simulation().radial_zones();
  const int kNumAngularZones = simulation().angular_zones();
  const double kRadialZoneWidth = 1 / max(1., kNumRadialZones - 1.);
  const double kAngularZoneWidth = 2 * M_PI / kNumAngularZones;
  const int kRows = isoplanatic_region->rows;
  const int kCols = isoplanatic_region->cols;

  *isoplanatic_region = 0;

  // Loop through every pixel, determining the interpolation weight for the
  // given region
  for (int i = 0; i < kRows; i++) {
    double y = i - 0.5 * kRows;
    for (int j = 0; j < kCols; j++) {
      double x = j - 0.5 * kCols;

      // Compute polar coordinates from the center of the image
      double r = sqrt(x*x + y*y) / (0.5 * max(kRows, kCols));
      double theta = atan2(y, x);
      while (theta < 0) theta += 2 * M_PI;

      // Find the two bounding radial regions
      double r_index = r / kRadialZoneWidth;
      int r_lt_index = min(max(int(floor(r_index)), 0), kNumRadialZones - 1);
      int r_gt_index = min(int(ceil(r_index)), kNumRadialZones - 1);

      // Compute r weight if we're in the given region
      if (r_lt_index != radial_idx && r_gt_index != radial_idx) continue;
      double r_blend = r_index - r_lt_index;
      double r_weight = 0;
      if (radial_idx == r_lt_index) r_weight += 1 - r_blend;
      if (radial_idx == r_gt_index) r_weight += r_blend;

      // If we're in the r=0 region, there is no off-axis wavefront error, so
      // there's no need to look at theta
      double theta_weight = 1;
      if (radial_idx > 0) {
        // Compute the two bounding angular region
        double theta_index = theta / kAngularZoneWidth;
        int theta_lt_index = max(int(floor(theta_index)), 0);
        int theta_gt_index = int(ceil(theta_index)) % kNumAngularZones;

        // Compute the theta weight if we're in the given region
        if (theta_lt_index != angular_idx && theta_gt_index != angular_idx)
          continue;
        theta_weight = 0;
        double theta_blend = theta_index - theta_lt_index;
        if (angular_idx == theta_lt_index) theta_weight += 1 - theta_blend;
        if (angular_idx == theta_gt_index) theta_weight += theta_blend;
      }

      (*isoplanatic_region)(i, j) = r_weight * theta_weight;
    }
  }
}

}
