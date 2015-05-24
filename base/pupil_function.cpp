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

  fftw_lock().lock();

  fftw_complex* pupil_func =
      (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
  fftw_complex* pupil_func_fft =
      (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);

  double* real_data = (double*) pupil_real_.data;
  double* imag_data = (double*) pupil_imag_.data;

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      const size_t index = i * cols + j;
      pupil_func[index][0] = real_data[index];
      pupil_func[index][1] = imag_data[index];
    }
  }

  fftw_plan fft_plan = fftw_plan_dft_2d(rows, cols, pupil_func, pupil_func_fft,
                                        FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_lock().unlock();

  fftw_execute(fft_plan);

  Mat psf(rows, cols, CV_64FC1);
  double* psf_data = (double*) psf.data;
  double psf_total = 0.0;
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      const size_t index = i * cols + j;
      const double real = pupil_func_fft[index][0];
      const double imag = pupil_func_fft[index][1];
      const double psf_val = real * real + imag * imag;

      psf_data[index] = psf_val;
      psf_total += psf_val;
    }
  }

  fftw_lock().lock();
  fftw_destroy_plan(fft_plan);
  fftw_free(pupil_func);
  fftw_free(pupil_func_fft);
  fftw_lock().unlock();

  psf /= psf_total;

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
