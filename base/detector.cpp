// File Description
// Author: Philip Salvaggio

#include "detector.h"

#include "base/math_utils.h"
#include "base/opencv_utils.h"
#include "io/logging.h"

#include <fftw3.h>

using namespace std;
using cv::Mat;

namespace mats {

Detector::Detector(const DetectorParameters& det_params,
                   const SimulationConfig& sim_params,
                   int sim_index)
    : det_params_(det_params),
      sim_params_(sim_params),
      sim_index_(sim_index) {}

// The governing equation for the response of a pixel in electrons is
//              pi * A_d * t_int * F
// S(x, y, l) = --------------------- * L(x, y, l) * tau(l) * QE(l) * l
//              h * c * (1 + 4(F#)^2)
//
// This equation assumes that L has already been degraded witht the system OTF.
void Detector::ResponseElectrons(const vector<Mat>& radiance,
                                 const vector<double>& wavelengths,
                                 vector<Mat>* electrons) {
  // Validate the input.
  if (!electrons) return;
  if (radiance.size() == 0) return;
  if (radiance[0].rows != rows() || radiance[0].cols != cols()) {
    mainLog() << "The given radiance images did not have the proper size."
              << endl;
    return;
  }

  const Simulation& simulation(sim_params_.simulation(sim_index_));

  // Calculate the key scalar quantities in the governing equation.
  double focal_length = sim_params_.altitude() *
                        det_params_.pixel_pitch() /
                        simulation.gsd();
  double f_number = focal_length / simulation.encircled_diameter();
  double det_area = det_params_.pixel_pitch() * det_params_.pixel_pitch();
  double fill_factor = simulation.fill_factor();
  double int_time = simulation.integration_time();
  const double h = 6.62606957e-34;  // [J*s]
  const double c = 299792458;  // [m/s]

  // Calculate the leading scalar factor in the governing equation.
  double scalar_factor = M_PI * det_area * int_time * fill_factor /
                         (h * c * (1 + 4 * f_number * f_number));

  // Interpolate the QE spectrum of the detector to match the input spectral
  // radiance bands.
  vector<double> qe;
  for (int i = 0; i < wavelengths.size(); i++) {
    double wave = wavelengths[i];
    
    int j;
    for (j = 0; j < det_params_.band_size(); j++) {
      double band_center = det_params_.band(j).center_wavelength();
      if (band_center >= wave) {
        if (j == 0) {
          qe.push_back(det_params_.band(j).quantum_efficiency());
        } else {
          double last_band_center = det_params_.band(j-1).center_wavelength();
          double blend = (wave - last_band_center) /
                         (band_center - last_band_center);
          qe.push_back(blend * det_params_.band(j).quantum_efficiency() +
              (1-blend) * det_params_.band(j-1).quantum_efficiency());
        }
        break;
      }

      if (j == det_params_.band_size()) {
        qe.push_back(det_params_.band(j-1).quantum_efficiency());
      }
    }
  }

  // We're just going to use a flat transmittance for the optics.
  const double kTransmittance = 0.9;
  vector<double> transmittances(wavelengths.size(), kTransmittance);
  mainLog() << "Using a flat transmittance of " << kTransmittance
            << " for the telescope optics." << endl;

  vector<Mat> high_res_electrons;
  for (size_t i = 0; i < wavelengths.size(); i++) {
    high_res_electrons.push_back(
        scalar_factor * radiance[i] * transmittances[i] * qe[i] *
        wavelengths[i]);
  }

  // Combine the high-spectral resolution electrons into the output bands
  // defined by the detector.
  for (size_t i = 0; i < det_params_.band_size(); i++) {
    electrons->push_back(Mat(rows(), cols(), CV_64FC1));
    double band_sigma = det_params_.band(i).fwhm() / 2.3548;
    
    for (size_t j = 0; j < wavelengths.size(); j++) {
      double weight = Gaussian1D(wavelengths[j],
                                 det_params_.band(i).center_wavelength(),
                                 band_sigma);
      if (weight > 1e-4) {
        electrons->at(i) += weight * high_res_electrons[j];
      }
    }
  }
}

Mat Detector::GetSamplingOtf() {
  const int kRows = rows();
  const int kCols = cols();
  const double kSize = std::max(kRows, kCols);
  const double kPixelPitch = det_params_.pixel_pitch();
  const double kPiPixelPitch = M_PI * kPixelPitch;

  vector<Mat> otf_planes;
  otf_planes.push_back(Mat::zeros(kRows, kCols, CV_64FC1));
  otf_planes.push_back(Mat::zeros(kRows, kCols, CV_64FC1));

  double* real_data = (double*) otf_planes[0].data;
  for (int r = 0; r < kRows; r++) {
    int y = std::min(r, kRows - r);
    double eta = (1 / kPixelPitch) * (y / kSize);
    double eta_sinc = (y == 0) ? 1 : sin(kPiPixelPitch * eta) /
                                     (kPiPixelPitch * eta);

    for (int c = 0; c < kSize; c++) {
      int x = std::min(c, kCols - c);
      double xi = (1 / kPixelPitch) * (x / kSize);
      double xi_sinc = (x == 0) ? 1 : sin(kPiPixelPitch * xi) /
                                      (kPiPixelPitch * xi);

      real_data[r*kCols + c] = eta_sinc * xi_sinc;
    }
  }

  Mat otf;
  merge(otf_planes, otf);
  return otf;
}

Mat Detector::GetSmearOtf(double x_velocity, double y_velocity) {
  const int kRows = rows();
  const int kCols = cols();
  const double kSize = std::max(kRows, kCols);
  const double kPixelPitch = det_params_.pixel_pitch();
  const double kIntTime =
      sim_params_.simulation(sim_index_).integration_time();

  double xi_coeff = M_PI * x_velocity * kIntTime;
  double eta_coeff = M_PI * y_velocity * kIntTime;
  //double xi_coeff = M_PI * x_velocity * kIntTime * kPixelPitch;
  //double eta_coeff = M_PI * y_velocity * kIntTime * kPixelPitch;

  vector<Mat> otf_planes;
  otf_planes.push_back(Mat::zeros(kRows, kCols, CV_64FC1));
  otf_planes.push_back(Mat::zeros(kRows, kCols, CV_64FC1));

  double* real_data = (double*) otf_planes[0].data;
  for (int r = 0; r < kRows; r++) {
    int y = std::min(r, kRows - r);
    double eta = (1 / kPixelPitch) * (y / kSize);
                            
    for (int c = 0; c < kCols; c++) {
      int x = std::min(c, kCols - c);
      double xi = (1 / kPixelPitch) * (x / kSize);

      double sinc_param = xi_coeff * xi + eta_coeff * eta;
      real_data[r*kCols + c] = (sinc_param == 0) ? 1 :
                               sin(sinc_param) / sinc_param;
    }            
  }              

  double max_val;
  minMaxIdx(otf_planes[0], NULL, &max_val);
  std::cout << "Max of smear MTF = " << max_val << std::endl;
  minMaxIdx(otf_planes[1], NULL, &max_val);
  std::cout << "Max of smear MTF (imag) = " << max_val << std::endl;

  Mat otf;       
  merge(otf_planes, otf);
  return otf;
}

Mat Detector::GetJitterOtf(double jitter_std_dev) {
  const int kRows = rows();
  const int kCols = cols();
  const int kNumTimesteps = 100;
  const double kIntTime =
      sim_params_.simulation(sim_index_).integration_time();

  double delta_freq = 1 / kIntTime;
  vector<double> x_offset, y_offset;

  Mat phase(2, kNumTimesteps, CV_64FC1);
  randn(phase, 0, 1);

  fftw_complex* jitter_spectrum = fftw_alloc_complex(kNumTimesteps);
  fftw_complex* jitter_instance = fftw_alloc_complex(kNumTimesteps);

  fftw_plan fft_plan = fftw_plan_dft_1d(kNumTimesteps, jitter_spectrum,
                                        jitter_instance, FFTW_BACKWARD,
                                        FFTW_ESTIMATE);

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

  fftw_free(jitter_spectrum);
  fftw_free(jitter_instance);

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
  dft(jitter_psf, otf, cv::DFT_COMPLEX_OUTPUT);
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

Mat Detector::GetNoisePattern() const {
  const double kTemperature = det_params_.temperature();
  const double kIntTime =
      sim_params_.simulation(sim_index_).integration_time();

  const double kRefTemp = det_params_.darkcurr_reference_temp();
  const double kRefRms = det_params_.darkcurr_reference_rms();
  const double kDoublingTemp = det_params_.darkcurr_doubling_temp();

  double dark_rms = sqrt(kIntTime * kRefRms * kRefRms * kIntTime *
                         pow(2, (kTemperature - kRefTemp) / kDoublingTemp));
  Mat dark_noise(rows(), cols(), CV_64FC1);
  randn(dark_noise, 0, dark_rms);

  Mat read_noise(rows(), cols(), CV_64FC1);
  randn(read_noise, 0, det_params_.read_rms());

  return dark_noise + read_noise;
}

}