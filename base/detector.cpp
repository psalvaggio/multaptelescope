// File Description
// Author: Philip Salvaggio

#include "detector.h"

#include "base/fftw_lock.h"
#include "base/math_utils.h"
#include "base/opencv_utils.h"
#include "base/photon_noise.h"
#include "io/logging.h"

#include <iostream>
#include <numeric>

#include <fftw3.h>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

namespace mats {

Detector::Detector(const DetectorParameters& det_params)
    : det_params_(det_params),
      qe_spectrums_() {
  qe_spectrums_.reserve(det_params_.band_size());
  for (int i = 0; i < det_params_.band_size(); i++) {
    const DetectorBandpass& band(det_params_.band(i));

    size_t qe_size = min(band.wavelength_size(),
                         band.quantum_efficiency_size());
    qe_spectrums_.push_back(vector<QESample>(qe_size));

    for (size_t j = 0; j < qe_size; j++) {
      qe_spectrums_[i][j].wavelength = band.wavelength(j);
      qe_spectrums_[i][j].qe = band.quantum_efficiency(j);
    }

    sort(qe_spectrums_[i].begin(), qe_spectrums_[i].end(),
         [] (const QESample& a, const QESample& b) {
           return a.wavelength < b.wavelength;
         });
  }
}

double Detector::gain() const {
  return det_params_.electrons_per_adu();
}

double Detector::full_well_capacity() const {
  return det_params_.full_well_capacity();
}

void Detector::GetQESpectrum(const vector<double>& wavelengths,
                             int band_index,
                             vector<double>* qe) const {
  if (band_index < 0 || band_index >= det_params_.band_size()) return;

  const vector<QESample>& qe_spectrum(qe_spectrums_[band_index]);

  // Interpolate the QE spectrum of the detector to match the input spectral
  // radiance bands.
  for (size_t i = 0; i < wavelengths.size(); i++) {
    double wave = wavelengths[i];
    
    size_t j;
    for (j = 0; j < qe_spectrum.size(); j++) {
      double ref_wave = qe_spectrum[j].wavelength;
      if (ref_wave >= wave) {
        if (j == 0) {
          qe->push_back(qe_spectrum[0].qe);
        } else {
          double last_ref_wave = qe_spectrum[j-1].wavelength;
          double blend = (wave - last_ref_wave) / (ref_wave - last_ref_wave);

          qe->push_back(blend * qe_spectrum[j].qe +
              (1-blend) * qe_spectrum[j-1].qe);
        }
        break;
      }
    }

    if (j == qe_spectrum.size()) {
      qe->push_back(qe_spectrum[j-1].qe);
    }
  }
}

double Detector::GetEffectiveQE(const vector<double>& wavelengths,
                                const vector<double>& spectral_weighting,
                                int band_index) const {
  vector<double> qe;
  GetQESpectrum(wavelengths, band_index, &qe);

  double eff_qe = 0;
  for (size_t i = 0; i < wavelengths.size(); i++) {
    eff_qe += spectral_weighting[i] * qe[i];
  }
  double total_weight = 
      accumulate(begin(spectral_weighting), end(spectral_weighting), 0.);

  return eff_qe / total_weight;
}

// The governing equation for the response of a pixel in electrons is
//              A_d * t_int
// S(x, y, l) = ----------- * E(x, y, l) *  QE(l) * l
//                h * c
//
// This equation assumes that L has already been degraded witht the system OTF
// and attenuated by the optics.
void Detector::ResponseElectrons(const vector<Mat>& irradiance,
                                 const vector<double>& wavelengths,
                                 double int_time,
                                 vector<Mat>* electrons) {
  // Validate the input.
  if (!electrons) return;
  if (irradiance.size() == 0) return;
  if (irradiance[0].rows != rows() || irradiance[0].cols != cols()) {
    mainLog() << "Warning: The given irradiance images were not the same size "
              << "as the detector." << endl;
  }

  //const Simulation& simulation(sim_params_.simulation(sim_index_));

  // Calculate the key scalar quantities in the governing equation.
  double det_area = det_params_.pixel_pitch() * det_params_.pixel_pitch() *
                    det_params_.fill_factor();
  //double int_time = simulation.integration_time();
  const double h = 6.62606957e-34;  // [J*s]
  const double c = 299792458;  // [m/s]

  // Calculate the leading scalar factor in the governing equation.
  double scalar_factor = det_area * int_time / (h * c);

  PhotonNoise photon_noise;
  vector<Mat> high_res_electrons;
  for (size_t i = 0; i < wavelengths.size(); i++) {
    high_res_electrons.push_back(
        scalar_factor * irradiance[i] * wavelengths[i]);
    photon_noise.AddPhotonNoise(&high_res_electrons[i]);
  }

  // Combine the high-spectral resolution electrons into the output bands
  // defined by the detector.
  AggregateSignal(high_res_electrons, wavelengths, false, electrons);

  for (int i = 0; i < det_params_.band_size(); i++) {
    electrons->at(i) += GetNoisePattern(int_time,
                                        electrons->at(i).rows,
                                        electrons->at(i).cols);
  }
}

void Detector::AggregateSignal(const vector<Mat>& signal,
                               const vector<double>& wavelengths,
                               bool normalize,
                               vector<Mat>* output) const {
  if (signal.size() == 0 || !output) return;

  const int kRows = signal[0].rows;
  const int kCols = signal[0].cols;
  const int kDataType = signal[0].type();

  double total_weight = 0;

  output->clear();
  for (int i = 0; i < det_params_.band_size(); i++) {
    output->push_back(Mat::zeros(kRows, kCols, kDataType));

    vector<double> qe;
    GetQESpectrum(wavelengths, i, &qe);
    
    for (size_t j = 0; j < wavelengths.size(); j++) {
      if (qe[j] > 1e-4) {
        output->at(i) += qe[j] * signal[j];
        total_weight += qe[j];
      }
    }

    if (normalize) {
      output->at(i) /= total_weight;
    }
  }
}

void Detector::Quantize(const vector<Mat>& electrons,
                        vector<Mat>* dig_counts) const {
  if (!dig_counts) return;
  dig_counts->clear();

  int data_type = -9999;
  int bit_depth = det_params_.a_d_bit_depth();
  if (bit_depth <= 8) {
    data_type = CV_8UC1;
  } else if (bit_depth <= 16) {
    data_type = CV_16UC1;
  } else if (bit_depth <= 31) {
    data_type = CV_32SC1;
  } else {
    mainLog() << "Error: Cannot support bit depths above 31.";
    return;
  }

  long max_dig_count = ((long) 1) << det_params_.a_d_bit_depth();
  double gain = 1 / det_params_.electrons_per_adu();

  for (size_t i = 0; i < electrons.size(); i++) {
    Mat tmp = gain * electrons[i];
    tmp = min(tmp, max_dig_count);
    tmp.convertTo(tmp, data_type);
    tmp.convertTo(tmp, CV_64FC1);
    dig_counts->push_back(tmp);
  }
}

Mat_<complex<double>> Detector::GetSamplingOtf(int rows, int cols) {
  const int kRows = (rows > 0) ? rows : this->rows();
  const int kCols = (cols > 0) ? cols : this->cols();
  const int kDetRows = this->rows();
  const int kDetCols = this->cols();

  Mat_<complex<double>> otf(kRows, kCols);

  for (int r = 0; r < kRows; r++) {
    int y = min(r, kRows - r);
    double eta = double(y) / kDetRows;  // [cyc / pixel]
    double pi_eta = M_PI * eta;
    double eta_sinc = (y == 0) ? 1 : sin(pi_eta) / pi_eta;

    for (int c = 0; c < kCols; c++) {
      int x = min(c, kCols - c);
      double xi = double(x) / kDetCols;
      double pi_xi = M_PI * xi;
      double xi_sinc = (x == 0) ? 1 : sin(pi_xi) / pi_xi;

      otf(r, c) = eta_sinc * xi_sinc;
    }
  }

  return otf;
}

Mat_<complex<double>> Detector::GetSmearOtf(double x_velocity,
                                            double y_velocity,
                                            double int_time,
                                            int rows,
                                            int cols) {
  const int kRows = (rows > 0) ? rows : this->rows();
  const int kCols = (cols > 0) ? cols : this->cols();
  const double kSize = max(kRows, kCols);
  const double kPixelPitch = det_params_.pixel_pitch();

  double xi_coeff = M_PI * x_velocity * int_time;
  double eta_coeff = M_PI * y_velocity * int_time;

  Mat_<complex<double>> otf(kRows, kCols);

  for (int r = 0; r < kRows; r++) {
    int y = min(r, kRows - r);
    double eta = (1 / kPixelPitch) * (y / kSize);
                            
    for (int c = 0; c < kCols; c++) {
      int x = min(c, kCols - c);
      double xi = (1 / kPixelPitch) * (x / kSize);

      double sinc_param = xi_coeff * xi + eta_coeff * eta;
      otf(r, c) = (sinc_param == 0) ? complex<double>(1, 0) :
                  complex<double>(sin(sinc_param) / sinc_param, 0);
    }            
  }              

  return otf;
}

Mat_<complex<double>> Detector::GetJitterOtf(double jitter_std_dev,
                                             double int_time,
                                             int rows,
                                             int cols) {
  const int kRows = (rows > 0) ? rows : this->rows();
  const int kCols = (cols > 0) ? cols : this->cols();
  const int kNumTimesteps = 100;

  double delta_freq = 1 / int_time;
  vector<double> x_offset, y_offset;

  Mat phase(2, kNumTimesteps, CV_64FC1);
  randn(phase, 0, 1);

  fftw_lock().lock();
  fftw_complex* jitter_spectrum = fftw_alloc_complex(kNumTimesteps);
  fftw_complex* jitter_instance = fftw_alloc_complex(kNumTimesteps);

  fftw_plan fft_plan = fftw_plan_dft_1d(kNumTimesteps, jitter_spectrum,
                                        jitter_instance, FFTW_BACKWARD,
                                        FFTW_ESTIMATE);
  fftw_lock().unlock();

  jitter_spectrum[0][0] = 0;
  jitter_spectrum[0][1] = 0;
  for (int i = 1; i < kNumTimesteps; i++) {
    double magnitude = 1 / sqrt(i * delta_freq);
    jitter_spectrum[i][0] = magnitude * cos(phase.at<double>(0, i));
    jitter_spectrum[i][1] = magnitude * sin(phase.at<double>(0, i));
  }

  fftw_execute(fft_plan);

  for (int i = 0; i < kNumTimesteps; i++) {
    x_offset.push_back(jitter_instance[i][1]);
  }

  for (int i = 1; i < kNumTimesteps; i++) {
    double magnitude = 1 / sqrt(i * delta_freq);
    jitter_spectrum[i][0] = magnitude * cos(phase.at<double>(1, i));
    jitter_spectrum[i][1] = magnitude * sin(phase.at<double>(1, i));
  }

  fftw_execute(fft_plan);

  for (int i = 0; i < kNumTimesteps; i++) {
    y_offset.push_back(jitter_instance[i][1]);
  }

  fftw_lock().lock();
  fftw_destroy_plan(fft_plan);
  fftw_free(jitter_spectrum);
  fftw_free(jitter_instance);
  fftw_lock().unlock();

  double mean_x = 0, mean_sq_x = 0;
  double mean_y = 0, mean_sq_y = 0;
  for (int i = 0; i < kNumTimesteps; i++) {
    mean_x += x_offset[i];
    mean_sq_x += x_offset[i] * x_offset[i];
    mean_y += y_offset[i];
    mean_sq_y += y_offset[i] * y_offset[i];
  }
  mean_x /= kNumTimesteps;
  mean_sq_x /= kNumTimesteps;
  mean_y /= kNumTimesteps;
  mean_sq_y /= kNumTimesteps;
  double std_x = sqrt(mean_sq_x - mean_x*mean_x);
  double std_y = sqrt(mean_sq_y - mean_y*mean_y);

  for (int i = 0; i < kNumTimesteps; i++) {
    x_offset[i] = (x_offset[i] - mean_x) / std_x * jitter_std_dev +
                  kCols / 2.0;
    y_offset[i] = (y_offset[i] - mean_y) / std_y * jitter_std_dev +
                  kRows / 2.0;
  }

  Mat jitter_psf = Mat::zeros(kRows, kCols, CV_64FC1);

  double* psf_data = (double*) jitter_psf.data;
  
  for (int i = 0; i < kNumTimesteps; i++) {
    int x_off_rnd = round(x_offset[i]);
    int y_off_rnd = round(y_offset[i]);

    if (x_off_rnd > x_offset[i]) {
      double x_blend = x_off_rnd - x_offset[i];

      if (y_off_rnd > y_offset[i]) {
        double y_blend = y_off_rnd - y_offset[i];

        psf_data[y_off_rnd * kCols + x_off_rnd] += 
          (1 - x_blend) * (1 - y_blend);
        psf_data[y_off_rnd * kCols + x_off_rnd - 1] += 
          x_blend * (1 - y_blend);
        psf_data[(y_off_rnd - 1) * kCols + x_off_rnd] += 
          (1 - x_blend) * (y_blend);
        psf_data[(y_off_rnd - 1) * kCols + x_off_rnd - 1] += 
          x_blend * y_blend;
      } else {
        double y_blend = y_offset[i] - y_off_rnd;

        psf_data[y_off_rnd * kCols + x_off_rnd] += 
          (1 - x_blend) * (1 - y_blend);
        psf_data[y_off_rnd * kCols + x_off_rnd - 1] += 
          x_blend * (1 - y_blend);
        psf_data[(y_off_rnd + 1) * kCols + x_off_rnd] += 
          (1 - x_blend) * (y_blend);
        psf_data[(y_off_rnd + 1) * kCols + x_off_rnd - 1] += 
          x_blend * y_blend;
      }
    } else {
      double x_blend = x_offset[i] - x_off_rnd;

      if (y_off_rnd > y_offset[i]) {
        double y_blend = y_off_rnd - y_offset[i];

        psf_data[y_off_rnd * kCols + x_off_rnd] += 
          (1 - x_blend) * (1 - y_blend);
        psf_data[y_off_rnd * kCols + x_off_rnd + 1] += 
          x_blend * (1 - y_blend);
        psf_data[(y_off_rnd - 1) * kCols + x_off_rnd] += 
          (1 - x_blend) * (y_blend);
        psf_data[(y_off_rnd - 1) * kCols + x_off_rnd + 1] += 
          x_blend * y_blend;
      } else {
        double y_blend = y_offset[i] - y_off_rnd;

        psf_data[y_off_rnd * kCols + x_off_rnd] += 
          (1 - x_blend) * (1 - y_blend);
        psf_data[y_off_rnd * kCols + x_off_rnd + 1] += 
          x_blend * (1 - y_blend);
        psf_data[(y_off_rnd + 1) * kCols + x_off_rnd] += 
          (1 - x_blend) * (y_blend);
        psf_data[(y_off_rnd + 1) * kCols + x_off_rnd + 1] += 
          x_blend * y_blend;
      }
    }
  }

  jitter_psf = FFTShift(jitter_psf);

  Mat otf;
  dft(jitter_psf, otf, DFT_COMPLEX_OUTPUT);
  vector<Mat> otf_planes;
  split(otf, otf_planes);

  Mat mtf;
  magnitude(otf_planes[0], otf_planes[1], mtf);
  double max_mtf;
  minMaxIdx(mtf, NULL, &max_mtf);
  otf_planes[0] = mtf / max_mtf;
  otf_planes[1] = Mat::zeros(mtf.rows, mtf.cols, CV_64FC1);
  merge(otf_planes, otf);

  return otf;
}

Mat Detector::GetNoisePattern(double int_time, int rows, int cols) const {
  const double kTemperature = det_params_.temperature();
  //const double kIntTime =
      //sim_params_.simulation(sim_index_).integration_time();

  const double kRefTemp = det_params_.darkcurr_reference_temp();
  const double kRefRms = det_params_.darkcurr_reference_rms();
  const double kDoublingTemp = det_params_.darkcurr_doubling_temp();

  //double dark_rms = kRefRms * kIntTime *
  double dark_rms = kRefRms * int_time *
                    pow(2, (kTemperature - kRefTemp) / kDoublingTemp);
  Mat dark_noise(rows, cols, CV_64FC1);
  randn(dark_noise, 0, dark_rms);

  Mat read_noise(rows, cols, CV_64FC1);
  randn(read_noise, 0, det_params_.read_rms());

  return dark_noise + read_noise;
}

}
