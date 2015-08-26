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
    : aperture_(ApertureFactory::Create(sim_config, sim_index)),
      detector_(new Detector(det_params, sim_config, sim_index)),
      parallelism_(false) {}

Telescope::~Telescope() {}

const SimulationConfig& Telescope::sim_params() const {
  return detector_->sim_params();
}

// In this model, the user specifies the pixel pitch of the detector, the
// altitude at which the telescope is flying, and the pixel size on the ground.
// Thus, the focal length must be a calculated parameter.
double Telescope::FocalLength() const {
  const mats::Simulation& sim = detector_->simulation();
  if (sim.has_focal_length()) return sim.focal_length();

  return detector_->sim_params().altitude() *
         detector_->det_params().pixel_pitch() /
         detector_->simulation().gsd();  // [m]
}

double Telescope::FNumber() const {
  return FocalLength() / aperture_->encircled_diameter();
}

double Telescope::GNumber(double lambda) const {
  double f_number = FNumber();

  vector<double> transmission;
  vector<double> wavelengths(1, lambda);
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
    double weight =
        spectral_weighting[max(i, spectral_weighting.size() - 1)];
    q += wavelengths[i] * fnumber * weight / p;
    total_weight += weight;
  }
  return q / total_weight;
}

void Telescope::Image(const vector<Mat>& radiance,
                      const vector<double>& wavelength,
                      vector<Mat>* image,
                      vector<Mat>* otf) {
  // Compute the OTF for each of the spectral points in the input data.
  vector<Mat> spectral_otfs;
  ComputeOtf(wavelength, &spectral_otfs);

  vector<Mat> blurred_irradiance(radiance.size());

  if (!parallelism_) {
    for (size_t i = 0; i < radiance.size(); i++) {
      DegradeImage(radiance[i], spectral_otfs[i], &(blurred_irradiance[i]));
      blurred_irradiance[i] /= GNumber(wavelength[i]);
    }
  } else {
    struct {
      const Telescope* self;
      const vector<Mat>* radiance;
      const vector<Mat>* spectral_otfs;
      const vector<double>* wavelength;
      vector<Mat>* degraded;
      void operator()(const tbb::blocked_range<int>& range) const {
        for (int i = range.begin(); i != range.end(); i++) {
          self->DegradeImage((*radiance)[i], (*spectral_otfs)[i],
                             &((*degraded)[i]));
          (*degraded)[i] /= self->GNumber((*wavelength)[i]);
        }
      }
    } worker{this, &radiance, &spectral_otfs, &wavelength, &blurred_irradiance};
    tbb::parallel_for(tbb::blocked_range<int>(0, wavelength.size()), worker);
  }
                     
  if (otf != NULL) {
    otf->clear();
    
    // Get the transmission spectrum of the optics.
    vector<double> transmittances;
    GetTransmissionSpectrum(wavelength, &transmittances);

    for (size_t i = 0; i < wavelength.size(); i++) {
      spectral_otfs[i] *= transmittances[i];
    }
    detector_->AggregateSignal(spectral_otfs, wavelength, true, otf);

    for (auto& tmp : *otf) tmp /= std::abs(tmp.at<complex<double>>(0, 0));
  }

  vector<Mat> electrons;
  //detector_->ResponseElectrons(blurred_irradiance, wavelength, &electrons);
  //detector_->Quantize(electrons, image);
  detector_->ResponseElectrons(blurred_irradiance, wavelength, image);
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
                           vector<Mat>* otf) const {
  const int kOtfSize = aperture_->params().array_size();

  vector<Mat> ap_otf;
  ComputeApertureOtf(wavelengths, &ap_otf);

  SystemOtf wave_invar_sys_otf;
  wave_invar_sys_otf.PushOtf(detector_->GetSmearOtf(0, 0, kOtfSize, kOtfSize));
  wave_invar_sys_otf.PushOtf(detector_->GetJitterOtf(0, kOtfSize, kOtfSize));
  Mat wave_invariant_otf = wave_invar_sys_otf.GetOtf();

  for (size_t i = 0; i < wavelengths.size(); i++) {
    SystemOtf sys_otf;
    sys_otf.PushOtf(ap_otf[i]);
    sys_otf.PushOtf(wave_invariant_otf);
    otf->push_back(sys_otf.GetOtf());
  }
}

void Telescope::ComputeEffectiveOtf(const vector<double>& wavelengths,
                                    const vector<double>& weights,
                                    cv::Mat* otf) const {
  if (!otf || wavelengths.size() == 0 || weights.size() < wavelengths.size()) {
    cerr << "Telescope::ComputeEffectiveOtf: Invalid Input" << endl;
    return;
  }

  vector<Mat> spectral_otf;
  ComputeOtf(wavelengths, &spectral_otf);

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
                                   vector<Mat>* otf) const {
  otf->clear();
  for (size_t i = 0; i < wavelengths.size(); i++) {
    otf->emplace_back();
  }

  // The OTF varies drastically with respect to wavelength. So, we will be
  // calculating an OTF for each spectral band in our input radiance data.
  if (!parallelism_) {
    for (size_t i = 0; i < wavelengths.size(); i++) {
      ComputeApertureOtf(wavelengths[i], &(otf->at(i)));
    }
  } else {
    struct ComputeApertureOtfWorker {
      const Telescope* self;
      const vector<double>* wavelengths;
      vector<Mat>* otf;
      void operator()(const tbb::blocked_range<int>& range) const {
        for (int i = range.begin(); i != range.end(); i++) {
          self->ComputeApertureOtf(wavelengths->at(i), &(otf->at(i)));
        }
      }
    };

    ComputeApertureOtfWorker worker;
    worker.self = this;
    worker.wavelengths = &wavelengths;
    worker.otf = otf;
    tbb::parallel_for(tbb::blocked_range<int>(0, wavelengths.size()), worker);
  }
}

void Telescope::ComputeApertureOtf(double wavelength, Mat* otf) const {
  // Array sizes
  const int kOtfSize = aperture_->params().array_size();

  // Get the aberrated pupil function from the aperture.
  PupilFunction pupil_func;
  aperture_->GetPupilFunction(wavelength, &pupil_func);

  // The coherent OTF is given by p[lamda * f * xi, lamda * f * eta],
  // where xi and eta are in [cyc/m]. The pixel pitch factor converts from
  // [cyc/pixel], which we want for degrading the image, and [cyc/m], which
  // is what is used by the pupil function.
  double pupil_scale = wavelength * FocalLength() / detector_->pixel_pitch();
    
  // Get the scale of the aperture, how many meters does each pixel represent
  // on the aperture plane [m/pix]
  double aperture_scale = pupil_func.meters_per_pixel();

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
           Size(kOtfSize, kOtfSize), 0, 0, INTER_NEAREST);
  }

  *otf = IFFTShift(scaled_otf);

  otf->at<complex<double>>(0, 0) /= std::abs(otf->at<complex<double>>(0, 0));
}

}
