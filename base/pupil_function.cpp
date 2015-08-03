// File Description
// Author: Philip Salvaggio

#include "pupil_function.h"

#include "base/fftw_lock.h"

#include <fftw3.h>
#include <vector>

using namespace cv;

namespace mats {

PupilFunction::PupilFunction()
    : pupil_real_(),
      pupil_imag_(),
      meters_per_pixel_(0) {}

PupilFunction::~PupilFunction() {}

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

Mat PupilFunction::PointSpreadFunction() {
  const size_t rows = pupil_real_.rows;
  const size_t cols = pupil_real_.cols;
  const size_t size = rows * cols;

  std::vector<Mat> pupil_planes{pupil_real_, pupil_imag_};
  Mat pupil, pupil_fft;
  merge(pupil_planes, pupil);
  dft(pupil, pupil_fft, DFT_COMPLEX_OUTPUT);

  std::vector<Mat> pupil_fft_planes;
  split(pupil_fft, pupil_fft_planes);

  Mat psf = pupil_fft_planes[0].mul(pupil_fft_planes[0]) +
            pupil_fft_planes[1].mul(pupil_fft_planes[1]);
  normalize(psf, psf, 1, 0, NORM_L1);

  return psf;
}

Mat PupilFunction::OpticalTransferFunction() {
  Mat psf = PointSpreadFunction();
  Mat otf;

  dft(psf, otf, DFT_COMPLEX_OUTPUT);
  return otf;
}

Mat PupilFunction::ModulationTransferFunction() {
  Mat psf = PointSpreadFunction();
  Mat otf;

  dft(psf, otf, DFT_COMPLEX_OUTPUT);

  std::vector<Mat> otf_planes;
  split(otf, otf_planes);

  Mat mtf;
  cv::magnitude(otf_planes[0], otf_planes[1], mtf);
  return mtf;
}

}
