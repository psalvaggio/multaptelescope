// File Description
// Author: Philip Salvaggio

#include "telescope.h"

#include "base/detector.h"
#include "base/opencv_utils.h"
#include "base/pupil_function.h"
#include "base/system_otf.h"
#include "io/logging.h"
#include "optical_designs/aperture.h"

#include <opencv/highgui.h>

#include <algorithm>

using std::vector;
using namespace cv;

namespace mats {

Telescope::Telescope(const SimulationConfig& sim_config,
                     int sim_index,
                     const DetectorParameters& det_params)
    : aperture_(ApertureFactory::Create(sim_config, sim_index)),
      detector_(new Detector(det_params, sim_config, sim_index)) {}

Telescope::~Telescope() {}

// In this model, the user specifies the pixel pitch of the detector, the
// altitude at which the telescope is flying, and the pixel size on the ground.
// Thus, the focal length must be a calculated parameter.
double Telescope::FocalLength() const {
  return detector_->sim_params().altitude() *
         detector_->det_params().pixel_pitch() /
         detector_->simulation().gsd();  // [m]
}

double Telescope::FNumber() {
  return FocalLength() / aperture_->encircled_diameter();
}

double Telescope::GNumber(double lambda) {
  double f_number = FNumber();

  vector<double> transmission;
  vector<double> wavelengths(1, lambda);
  GetTransmissionSpectrum(wavelengths, &transmission);

  return (1 + 4 * f_number * f_number) /
         (M_PI * transmission[0] * aperture_->fill_factor());
}

void Telescope::Image(const std::vector<Mat>& radiance,
                      const std::vector<double>& wavelength,
                      std::vector<Mat>* image,
                      std::vector<Mat>* otf) {
  // Compute the OTF for each of the spectral points in the input data.
  vector<Mat> spectral_otfs;
  ComputeOtf(wavelength, &spectral_otfs);

  // Get the transmission spectrum of the optics.
  vector<double> transmittances;
  GetTransmissionSpectrum(wavelength, &transmittances);

  vector<Mat> blurred_irradiance;
  for (size_t i = 0; i < radiance.size(); i++) {
    Mat img_fft;
    dft(radiance[i], img_fft, DFT_COMPLEX_OUTPUT);

    Mat blurred_fft;
    if (img_fft.rows != spectral_otfs[i].rows ||
        img_fft.cols != spectral_otfs[i].cols) {
      Mat scaled_spectrum;
      cv::resize(spectral_otfs[i], scaled_spectrum, img_fft.size());

      cv::mulSpectrums(img_fft, scaled_spectrum, blurred_fft, 0);
    } else {
      cv::mulSpectrums(img_fft, spectral_otfs[i], blurred_fft, 0);
    }

    Mat tmp_blurred_irradiance;
    dft(blurred_fft, tmp_blurred_irradiance,
        DFT_INVERSE | DFT_SCALE | DFT_REAL_OUTPUT);
    tmp_blurred_irradiance /= GNumber(wavelength[i]);

    blurred_irradiance.push_back(tmp_blurred_irradiance);
    spectral_otfs[i] *= transmittances[i];
  }

  if (otf != NULL) {
    otf->clear();
    detector_->AggregateSignal(spectral_otfs, wavelength, true, otf);
    for (size_t i = 0; i < otf->size(); i++) {
      vector<Mat> tmp_otf_planes;
      split((*otf)[i], tmp_otf_planes);
      double norm = sqrt(tmp_otf_planes[0].at<double>(0, 0) * 
                         tmp_otf_planes[0].at<double>(0, 0) +
                         tmp_otf_planes[1].at<double>(0, 0) * 
                         tmp_otf_planes[1].at<double>(0, 0));
      tmp_otf_planes[0] /= norm;
      tmp_otf_planes[1] /= norm;
      merge(tmp_otf_planes, (*otf)[i]);
    }
  }

  vector<Mat> electrons;
  detector_->ResponseElectrons(blurred_irradiance, wavelength, &electrons);
  detector_->Quantize(electrons, image);
}

void Telescope::ComputeOtf(const vector<double>& wavelengths,
                           std::vector<Mat>* otf) {
  vector<Mat> ap_otf;
  ComputeApertureOtf(wavelengths, &ap_otf);

  SystemOtf wave_invar_sys_otf;
  wave_invar_sys_otf.PushOtf(detector_->GetSamplingOtf());
  wave_invar_sys_otf.PushOtf(
      detector_->GetSmearOtf(0, detector_->pixel_pitch() * 2.5e4));
  wave_invar_sys_otf.PushOtf(detector_->GetJitterOtf(0.1));
  Mat wave_invariant_otf = wave_invar_sys_otf.GetOtf();

  vector<Mat> otf_planes;
  Mat mtf;
  split(wave_invariant_otf, otf_planes);
  magnitude(otf_planes[0], otf_planes[1], mtf);

  for (size_t i = 0; i < wavelengths.size(); i++) {
    SystemOtf sys_otf;
    sys_otf.PushOtf(ap_otf[i]);
    sys_otf.PushOtf(wave_invariant_otf);
    otf->push_back(sys_otf.GetOtf());
  }
}

void Telescope::GetTransmissionSpectrum(
    const std::vector<double>& wavelengths,
    std::vector<double>* transmission) const {
  if (!transmission) return;

  // We're just going to use a flat transmittance for the optics.
  const double kTransmittance = 0.9;
  transmission->clear();
  transmission->resize(wavelengths.size(), kTransmittance);
  mainLog() << "Using a flat transmittance of " << kTransmittance
            << " for the telescope optics." << std::endl;
}

void Telescope::ComputeApertureOtf(const vector<double>& wavelengths,
                                   vector<Mat>* otf) {
  // Array sizes
  const int kOtfSize = aperture_->params().array_size();
  const int kNumRows = detector_->rows();
  const int kNumCols = detector_->cols();

  Mat aperture_wfe = aperture_->GetWavefrontError();

  /*
  VideoWriter output_vid;
  output_vid.open("/Users/philipsalvaggio/Desktop/mtf.avi",
                  CV_FOURCC('M','J','P','G'), 30, Size(aperture_wfe.rows, aperture_wfe.cols), true);
  */

  // The OTF varies drastically with respect to wavelength. So, we will be
  // calculating an OTF for each spectral band in our input radiance data.
  for (size_t i = 0; i < wavelengths.size(); i++) {
    // Get the aberrated pupil function from the aperture.
    PupilFunction pupil_func;
    aperture_->GetPupilFunction(aperture_wfe, wavelengths[i], &pupil_func);

    // The coherent OTF is given by p[lamda * f * xi, lamda * f * eta],
    // where xi and eta are in [cyc/m]. The pixel pitch factor converts from
    // [cyc/pixel], which we want for degrading the image, and [cyc/m], which
    // is what is used by the pupil function.
    double pupil_scale = wavelengths[i] * FocalLength() /
                         detector_->pixel_pitch();
    
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
      // std::cout << "Scaling up " << (double)kOtfSize /
          // (otf_range.end - otf_range.start) << std::endl;
      resize(unscaled_otf(otf_range, otf_range), scaled_otf,
             Size(kNumRows, kNumCols), 0, 0, INTER_NEAREST);
    }

    otf->push_back(FFTShift(scaled_otf));
  }
}

}
