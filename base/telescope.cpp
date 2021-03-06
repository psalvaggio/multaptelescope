// File Description
// Author: Philip Salvaggio

#include "telescope.h"

#include "base/assertions.h"
#include "base/detector.h"
#include "base/opencv_utils.h"
#include "base/pupil_function.h"
#include "base/str_utils.h"
#include "base/system_otf.h"
#include "io/logging.h"
#include "optical_designs/aperture.h"
#include "deconvolution/constrained_least_squares.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <tbb/parallel_for.h>

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
      nonmodeled_mtf_(),
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


double Telescope::EffectiveQ(const vector<double>& wavelengths,
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
                      vector<Mat>* image) const {
  CHECK(!radiance.empty());
  CHECK(wavelength.size() == radiance.size());

  Rect roi;
  GetImagingRegion(radiance[0], &roi);
  CHECK(roi.width > 0);

  // Take the DFT of the given image.
  vector<Mat_<complex<double>>> radiance_dft(radiance.size());
  auto RadianceDft = [&roi, &radiance, &radiance_dft](size_t i) {
    dft(radiance[i](roi), radiance_dft[i], DFT_COMPLEX_OUTPUT);
  };
  if (parallelism_) {
    tbb::parallel_for(size_t{0}, radiance.size(), RadianceDft);
  } else {
    for (size_t i = 0; i < radiance.size(); i++) RadianceDft(i);
  }
    
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
      blurred_irradiance[i] = Mat::zeros(radiance_dft[i].size(), CV_64F);
    }

    // Loop through each isoplanatic region
    Mat_<double> isoplanatic_region(radiance_dft[0].size());
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
          OtfDegrade(radiance_dft[k], spectral_otfs[k], &degraded);
          multiply(degraded, isoplanatic_region, degraded,
              1 / GNumber(wavelength[k]));
          blurred_irradiance[k] += degraded;
        }

        if (i == 0) j = kAngularZones;
      }
    }
  } else {
    ImageInIsoplanaticRegion(wavelength, radiance_dft, 0, 0,
                             &blurred_irradiance);
  }
  radiance_dft.clear();
                     
  // Compute the detector response
  vector<Mat> electrons;
  double int_time = simulation().integration_time();
  detector_->ResponseElectrons(blurred_irradiance, wavelength, int_time,
                               &electrons);
  detector_->Quantize(electrons, image);
}


void Telescope::OtfDegrade(const Mat& radiance_dft,
                           const Mat& spectral_otf,
                           Mat* degraded) const {
  Mat blurred_dft;

  Mat otf(radiance_dft.size(), CV_64FC2);
  otf.setTo(Scalar(0, 0));
  int row_start = radiance_dft.rows / 2 - detector_->rows() / 2;
  int col_start = radiance_dft.cols / 2 - detector_->cols() / 2;
  Range row_range(row_start, row_start + detector_->rows()),
        col_range(col_start, col_start + detector_->cols());

  resize(FFTShift(spectral_otf), otf(row_range, col_range),
         Size(detector_->cols(), detector_->rows()));

  mulSpectrums(radiance_dft, IFFTShift(otf), blurred_dft, 0);

  dft(blurred_dft, *degraded, DFT_INVERSE | DFT_SCALE | DFT_REAL_OUTPUT);

  const int kRows = detector_->rows();
  const int kCols = detector_->cols();
  if (degraded->cols != kCols) {
    resize(*degraded, *degraded, Size(kCols, kRows), 0, 0, INTER_AREA);
  }
}


void Telescope::GetImagingRegion(const Mat& radiance, Rect* roi) const {
  const int kRows = detector_->rows();
  const int kCols = detector_->cols();
  const double kAspectRatio = double(kRows) / kCols;

  roi->x = 0;
  roi->y = 0;
  roi->width = 0;
  roi->height = 0;

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

  roi->width = cols;
  roi->height = rows;
}


void Telescope::Restore(const Mat_<double>& raw_image,
                        const vector<double>& wavelengths,
                        const vector<double>& illumination,
                        int band,
                        double smoothness,
                        Mat_<double>* restored) const {
  CHECK(restored);
  CHECK(!wavelengths.empty() && wavelengths.size() == illumination.size());

  vector<double> transmission, qe;
  GetTransmissionSpectrum(wavelengths, &transmission);
  detector_->GetQESpectrum(wavelengths, band, &qe);

  vector<double> weights(wavelengths.size());
  for (size_t i = 0; i < wavelengths.size(); i++) {
    weights[i] = transmission[i] * qe[i] * illumination[i];
  }

  ConstrainedLeastSquares cls;

  if (aperture_->HasOffAxisAberration()) {
    restored->create(raw_image.size());
    *restored = 0;

    const int kRadialZones = simulation().radial_zones();
    const int kAngularZones = simulation().angular_zones();
    const double kRadialZoneWidth = 1 / max(1., kRadialZones - 1.);
    const double kAngularZoneWidth = 2 * M_PI / kAngularZones;

    Mat_<double> isoplanatic_region(raw_image.size());
    for (int i = 0; i < kRadialZones; i++) {
      for (int j = 0; j < kAngularZones; j++) {
        cout << "Processing zone r " << (i+1) << "/" << kRadialZones
             << ", theta " << (j+1) << "/" << kAngularZones << endl;

        // Compute the interpolation weights for this region
        IsoplanaticRegion(i, j, &isoplanatic_region);

        // Compute the spactral OTF in the current isoplanatic region
        Mat_<complex<double>> otf;
        EffectiveOtf(wavelengths, weights, i * kRadialZoneWidth,
            j * kAngularZoneWidth, &otf);

        Mat_<double> restored_region;
        cls.Deconvolve(raw_image, otf, smoothness, &restored_region);

        *restored += isoplanatic_region.mul(restored_region);

        if (i == 0) j = kAngularZones;
      }
    }
  } else {
    Mat_<complex<double>> otf;
    EffectiveOtf(wavelengths, weights, 0, 0, &otf);
    cls.Deconvolve(raw_image, otf, smoothness, restored);
  }
}


void Telescope::Restore(const Mat_<double>& raw_image,
                        const vector<double>& wavelengths,
                        const vector<Mat>& illumination,
                        int band,
                        double smoothness,
                        Mat_<double>* restored) const {
  CHECK(restored);
  CHECK(!wavelengths.empty() && wavelengths.size() == illumination.size());

  vector<double> transmission, qe;
  GetTransmissionSpectrum(wavelengths, &transmission);
  detector_->GetQESpectrum(wavelengths, band, &qe);

  vector<Mat> spectral_otf;
  ComputeOtf(wavelengths, 0, 0, &spectral_otf);

  ConstrainedLeastSquares cls;
  
  restored->create(raw_image.size());
  *restored = 0;
  for (int r = 0; r < illumination[0].rows; r++) {
    for (int c = 0; c < illumination[0].cols; c++) {
      vector<double> spec_weights(wavelengths.size(), 0);
      for (size_t i = 0; i < wavelengths.size(); i++) {
        spec_weights[i] = qe[i] * transmission[i] *
                          illumination[i].at<double>(r, c);
      }
      Mat_<complex<double>> eff_otf;
      EffectiveOtf(spec_weights, spectral_otf, &eff_otf);

      Mat_<double> restored_tile;
      cls.Deconvolve(raw_image, eff_otf, smoothness, &restored_tile);

      Mat_<double> interp_weights(restored_tile.size());
      GridRegion(interp_weights, c, r,
                 illumination[0].cols, illumination[0].rows);
      *restored += restored_tile.mul(interp_weights);
    }
    imshow("Restored", ByteScale(*restored));
    waitKey(1);
  }
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
  if (!nonmodeled_mtf_.empty()) {
    vector<double> freq(nonmodeled_mtf_.size());
    for (size_t i = 0; i < nonmodeled_mtf_.size(); i++) {
      freq[i] = double(i) / (nonmodeled_mtf_.size() - 1);
    }

    Mat_<double> nonmod_mtf(ap_otf[0].size());
    double mtf_spacing = 1 / (nonmodeled_mtf_.size() - 1);
    for (int i = 0; i < nonmod_mtf.rows; i++) {
      double y = min(i, nonmod_mtf.rows - i) / double(nonmod_mtf.rows);
      for (int j = 0; j < nonmod_mtf.cols; j++) {
        double x = min(j, nonmod_mtf.cols - j) / double(nonmod_mtf.cols);
        double r = sqrt(x*x + y*y);

        auto lower = lower_bound(begin(freq), end(freq), r);
        if (lower == begin(freq)) {
          nonmod_mtf(i, j) = 1;
        } else if (lower == end(freq)) {
          nonmod_mtf(i, j) = nonmodeled_mtf_.back();
        } else {
          int gt_index = lower - begin(freq);
          int lt_index = gt_index - 1;
          double range =
              nonmodeled_mtf_[gt_index] - nonmodeled_mtf_[lt_index];

          double blend = (r - freq[lt_index]) / range;

          nonmod_mtf(i, j) = (1 - blend) * nonmodeled_mtf_[lt_index] +
                             blend * nonmodeled_mtf_[gt_index];
        }
      }
    }
    wave_invar_sys_otf.PushOtf(nonmod_mtf);
  }
  
  Mat_<complex<double>> wave_invariant_otf = wave_invar_sys_otf.GetOtf();

  otf->resize(wavelengths.size());
  auto ComputeOtfBody = [&] (size_t i) {
    mulSpectrums(ap_otf[i], wave_invariant_otf, (*otf)[i], 0);
  };

  if (parallelism_) {
    tbb::parallel_for(size_t{0}, wavelengths.size(), ComputeOtfBody);
  } else {
    for (size_t i = 0; i < wavelengths.size(); i++) ComputeOtfBody(i);
  }
}


void Telescope::EffectiveOtf(const vector<double>& wavelengths,
                             const vector<double>& weights,
                             double image_height,
                             double angle,
                             cv::Mat_<complex<double>>* otf) const {
  if (!otf || wavelengths.size() == 0 || weights.size() < wavelengths.size()) {
    cerr << "Telescope::EffectiveOtf: Invalid Input" << endl;
    return;
  }

  vector<Mat> spectral_otf;
  ComputeOtf(wavelengths, image_height, angle, &spectral_otf);

  EffectiveOtf(weights, spectral_otf, otf);
}

void Telescope::EffectiveOtf(const vector<double>& weights,
                             const std::vector<cv::Mat>& spectral_otf,
                             cv::Mat_<complex<double>>* otf) const {
  double total_weight = accumulate(begin(weights), end(weights), 0.0);

  spectral_otf[0].copyTo(*otf);
  (*otf) *= weights[0] / total_weight;
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


void Telescope::GridRegion(cv::Mat_<double>& weights,
                           int x_region, int y_region,
                           int x_regions, int y_regions) const {
  const double kYStep = 1. / y_regions;
  const double kXStep = 1. / x_regions;

  vector<double> x_weights(weights.cols, 0);
  for (int x = 0; x < weights.cols; x++) {
    double x_frac = (x / (weights.cols - 1.)) - 0.5 * kXStep;
    x_frac = max(x_frac, 0.);
    int next_x = ceil(x_frac * x_regions);

    double x_weight = 0;
    if (next_x == x_region) {
      x_weight = (next_x == 0) ? 1 : 1 - (next_x * kXStep - x_frac) / kXStep;
    } else if (next_x == x_region + 1) {
      x_weight = (next_x == x_regions) ? 1 :
                 (next_x * kXStep - x_frac) / kXStep;
    }
    x_weights[x] = x_weight;
  }

  for (int y = 0; y < weights.rows; y++) {
    double y_frac = (y / (weights.rows - 1.)) - 0.5 * kYStep;
    y_frac = max(y_frac, 0.);
    int next_y = ceil(y_frac * y_regions);

    double y_weight = 0;
    if (next_y == y_region) {
      y_weight = (next_y == 0) ? 1 : 1 - (next_y * kYStep - y_frac) / kYStep;
    } else if (next_y == y_region + 1) {
      y_weight = next_y == y_regions ? 1 :
                 (next_y * kYStep - y_frac) / kYStep;
    }
        
    for (int x = 0; x < weights.cols; x++) {
      weights(y, x) = y_weight * x_weights[x];
    }
  }
}


void Telescope::ImageInIsoplanaticRegion(
      const vector<double>& wavelength,
      const vector<Mat_<complex<double>>>& input_dft,
      int radial_zone,
      int angular_zone,         
      vector<Mat>* output) const {
  const int kRadialZones = simulation().radial_zones();
  const int kAngularZones = simulation().angular_zones();
  const double kRadialZoneWidth = 1 / max(1., kRadialZones - 1.);
  const double kAngularZoneWidth = 2 * M_PI / kAngularZones;

  // Compute the spactral OTF in the current isoplanatic region
  vector<Mat> spectral_otfs;
  ComputeOtf(wavelength,
             radial_zone * kRadialZoneWidth,
             angular_zone * kAngularZoneWidth,
             &spectral_otfs);

  // Perform image degradation and add to the total blurred image
  output->resize(input_dft.size());
  auto DegradeBody = [&] (size_t i) {
    OtfDegrade(input_dft[i], spectral_otfs[i], &((*output)[i]));
    (*output)[i] *= 1 / GNumber(wavelength[i]);
  };
  if (parallelism_) {
    tbb::parallel_for(size_t{0}, input_dft.size(), DegradeBody);
  } else {
    for (size_t i = 0; i < input_dft.size(); i++) DegradeBody(i);
  }
}

}
