// File Description
// Author: Philip Salvaggio

#include "pupil_function.h"

#include "base/fftw_lock.h"

#include <fftw3.h>
#include <vector>

using namespace std;
using namespace cv;

namespace mats {

PupilFunction::PupilFunction(int size, double reference_wavelength)
    : reference_wavelength_(reference_wavelength),
      pupil_real_(Mat_<double>::zeros(size, size)),
      pupil_imag_(Mat_<double>::zeros(size, size)),
      meters_per_pixel_(0) {}

PupilFunction::PupilFunction(PupilFunction&& other)
    : reference_wavelength_(other.reference_wavelength_),
      pupil_real_(move(other.pupil_real_)),
      pupil_imag_(move(other.pupil_imag_)),
      meters_per_pixel_(other.meters_per_pixel_) {}

PupilFunction::~PupilFunction() {}

PupilFunction& PupilFunction::operator=(PupilFunction&& other) {
  if (this == &other) return *this;

  reference_wavelength_ = other.reference_wavelength_;
  pupil_real_ = move(other.pupil_real_);
  pupil_imag_ = move(other.pupil_imag_);
  meters_per_pixel_ = other.meters_per_pixel_;
  return *this;
}

Mat PupilFunction::magnitude() const {
  Mat mag;
  cv::magnitude(pupil_real_, pupil_imag_, mag);
  return mag;
}

Mat PupilFunction::phase() const {
  Mat phase;
  cv::phase(pupil_real_, pupil_imag_, phase);
  return phase;
}

Mat_<double> PupilFunction::PointSpreadFunction() {
  vector<Mat> pupil_planes{pupil_real_, pupil_imag_};
  Mat pupil, pupil_fft;
  merge(pupil_planes, pupil);
  dft(pupil, pupil_fft, DFT_COMPLEX_OUTPUT);

  vector<Mat> pupil_fft_planes;
  split(pupil_fft, pupil_fft_planes);

  Mat_<double> psf = pupil_fft_planes[0].mul(pupil_fft_planes[0]) +
                     pupil_fft_planes[1].mul(pupil_fft_planes[1]);
  normalize(psf, psf, 1, 0, NORM_L1);

  return psf;
}

Mat_<complex<double>> PupilFunction::OpticalTransferFunction() {
  Mat_<double> psf = PointSpreadFunction();
  Mat_<complex<double>> otf;

  dft(psf, otf, DFT_COMPLEX_OUTPUT);
  return otf;
}

Mat_<double> PupilFunction::ModulationTransferFunction() {
  Mat_<complex<double>> otf = OpticalTransferFunction();
  vector<Mat> otf_planes;
  split(otf, otf_planes);

  Mat_<double> mtf;
  cv::magnitude(otf_planes[0], otf_planes[1], mtf);
  return mtf;
}

}
