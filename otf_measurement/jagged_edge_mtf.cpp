// File Description
// Author: Philip Salvaggio

#include "jagged_edge_mtf.h"

#include "base/opencv_utils.h"
#include "otf_measurement/hamming_window.h"
#include "otf_measurement/slant_edge_mtf.h"
#include "ransac/ransac.h"
#include "ransac/ransac_fit_line.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>

#include <fftw3.h>

using namespace std;
using namespace cv;

JaggedEdgeMtf::JaggedEdgeMtf() {
  gp_ = new Gnuplot();
  local_gp_ = true;
}

JaggedEdgeMtf::~JaggedEdgeMtf() {
  if (local_gp_ && gp_) {
    *gp_ << std::endl;
    delete gp_;
  }
}

void JaggedEdgeMtf::Analyze(const cv::Mat& image,
                            double* orientation,
                            std::vector<double>* mtf) {
  if (!orientation || !mtf) return;

  offset_t edge_offsets;

  // Detect the edge in the image.
  double edge[3];
  DetectEdge(image, edge, &edge_offsets);
  *orientation = atan2(-edge[0], edge[1]);
  if (*orientation >= M_PI / 2) *orientation -= M_PI;
  if (*orientation < -M_PI / 2) *orientation += M_PI;

  //imshow("Detected Edge", OverlayLine(image, edge));

  int num_bins = GetSamplesPerPixel(image, edge);

  // Compute and smooth the edge spread function (ESF)
  std::vector<double> esf, esf_stddev;
  GenerateEsf(image, edge, edge_offsets, num_bins, &esf, &esf_stddev);
  PlotEsf(esf, esf_stddev);
  SmoothEsf(&esf);

  fftw_complex* lsf = fftw_alloc_complex(esf.size());
  fftw_complex* otf = fftw_alloc_complex(esf.size());
  double max_lsf = 0;
  int max_idx = 0;
  lsf[0][0] = 0; lsf[0][1] = 0;
  for (size_t i = 1; i < esf.size(); i++) {
    lsf[i][0] = esf[i] - esf[i-1];
    lsf[i][1] = 0;
    if (fabs(lsf[i][0]) > max_lsf) {
      max_lsf = fabs(lsf[i][0]);
      max_idx = i;
    }
  }

  // Apply a Hamming window.
  double* hamming = new double[esf.size()];
  HammingWindow(esf.size(), max_idx, hamming);
  for (size_t i = 0; i < esf.size(); i++) {
    //lsf[i][0] *= hamming[i];
  }
  delete[] hamming;

  fftw_plan fft_plan  = fftw_plan_dft_1d(esf.size(), lsf, otf,
                                         FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(fft_plan);

  fftw_destroy_plan(fft_plan);
  fftw_free(lsf);

  fftw_complex* blur = fftw_alloc_complex(esf.size());
  fftw_complex* blur_otf = fftw_alloc_complex(esf.size());
  for (size_t i = 0; i < esf.size(); i++) {
    blur[i][0] = (i < 3) ? 1 / 3.0 : 0;
    blur[i][1] = 0;
  }
  fft_plan  = fftw_plan_dft_1d(esf.size(), blur, blur_otf,
                               FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(fft_plan);

  fftw_destroy_plan(fft_plan);
  fftw_free(blur);

  size_t nyquist_idx = esf.size() / (2*num_bins);
  //size_t nyquist_idx = esf.size() / (num_bins);

  double peak_mtf = sqrt(otf[0][0]*otf[0][0] + otf[0][1]*otf[0][1]);
  double peak_blur_mtf = sqrt(blur_otf[0][0]*blur_otf[0][0] +
                              blur_otf[0][1]*blur_otf[0][1]);
  mtf->push_back(1);
  for (size_t i = 1; i <= nyquist_idx; i++) {
    double positive_mag = sqrt(otf[i][0]*otf[i][0] + otf[i][1]*otf[i][1]);
    double negative_mag = sqrt(otf[esf.size()-i][0] * otf[esf.size()-i][0] +
                               otf[esf.size()-i][1] * otf[esf.size()-i][1]);
    double blur_pos_mag = sqrt(blur_otf[i][0]*blur_otf[i][0] +
                               blur_otf[i][1]*blur_otf[i][1]);
    double blur_neg_mag = sqrt(std::pow(blur_otf[esf.size()-i][0], 2) +
                               std::pow(blur_otf[esf.size()-i][1], 2));
    double blur_mag = 0.5 * (blur_pos_mag + blur_neg_mag) / peak_blur_mtf;
    mtf->push_back(0.5*(positive_mag + negative_mag) / peak_mtf / blur_mag);
  }

  fftw_free(otf);
  fftw_free(blur_otf);
}

cv::Mat JaggedEdgeMtf::OverlayLine(const cv::Mat& image,
                                   const double* line,
                                   const offset_t& offsets) {
  cv::Mat_<uint8_t> byte_scaled_image;
  ByteScale(image, byte_scaled_image);

  double lx = -line[1], ly = line[0];
  if (ly < 0) {
    lx *= -1; ly *= -1;
  }

  // Replicate into RGB planes.
  std::vector<cv::Mat> rgb_planes;
  for (int i = 0; i < 3; i++) {
    rgb_planes.push_back(cv::Mat());
    byte_scaled_image.copyTo(rgb_planes[i]);
  }
  uint8_t* b = rgb_planes[0].data;
  uint8_t* g = rgb_planes[1].data;
  uint8_t* r = rgb_planes[2].data;

  // Draw the line.
  for (const auto& offset_pt : offsets) {
    //int x = round(get<0>(offset_pt) * lx + get<1>(offset_pt) * line[0]) - 5;
    int x = round(get<0>(offset_pt) * lx + get<1>(offset_pt) * line[0]); 
    int y = round(get<0>(offset_pt) * ly + get<1>(offset_pt) * line[1]);
    r[y*image.cols + x] = 255 * (1 - get<1>(offset_pt));
    g[y*image.cols + x] = 255 * get<1>(offset_pt);
    b[y*image.cols + x] = 0;
  }

  cv::Mat rgb;
  cv::merge(rgb_planes, rgb);
  return rgb;
}

void JaggedEdgeMtf::SetGnuplot(Gnuplot* gp) {
  if (gp_ == gp) return;

  if (local_gp_ && gp_) {
    delete gp_;
  }

  gp_ = gp;
  local_gp_ = false;
}

bool JaggedEdgeMtf::DetectEdge(const cv::Mat& image,
                               double* edge,
                               offset_t* edge_offsets) {
  if (!edge) return false;

  cout << "Trying to detected edge." << endl;
  SlantEdgeMtf slant_edge;
  if (!slant_edge.DetectEdge(image, edge)) {
    return false;
  }
  cout << "Detected Edge." << endl;

  bool vertical_edge = abs(edge[0]) > abs(edge[1]);

  double lx = -edge[1];
  double ly = edge[0];
  if (ly < 0) {
    lx *= -1; ly *= -1;
  }

  cout << "Is vertical edge? " << vertical_edge << endl;

  // Take the x- and y-derivatives of the image.
  cv::Mat dy, dx;
  cv::absdiff(image,
              circshift(image, cv::Point2f(-1, 0), cv::BORDER_REFLECT), dx);
  cv::absdiff(image,
              circshift(image, cv::Point2f(0, -1), cv::BORDER_REFLECT), dy);
  dy.convertTo(dy, CV_64FC1);
  dx.convertTo(dx, CV_64FC1);
  cv::Mat_<double> deriv = dy.mul(dy) + dx.mul(dx);

  const int kUpperBound = vertical_edge ? image.rows : image.cols;
  const int kSearchHalfSize = 11;
  double average_line_spread = 0;
  for (int i = 0; i < kUpperBound; i++) {
    cv::Mat_<double> line = vertical_edge ? deriv.row(i) : deriv.col(i);
    double expected_pos = vertical_edge ? - (edge[1] * i - edge[2]) / edge[0]
                                        : - (edge[0] * i - edge[2]) / edge[1]; 

    cv::Range fit_region(
        max(0, (int)floor(expected_pos - kSearchHalfSize)),
        min((int)ceil(expected_pos + kSearchHalfSize + 1), kUpperBound));

    vector<double> search_region;
    double deriv_total = 0;
    for (int j = fit_region.start; j < fit_region.end; j++) {
      search_region.push_back(line(j));
      deriv_total += line(j);
    }
    for (auto& tmp : search_region) tmp /= deriv_total;

    double deriv_peak = 0, deriv_stddev = 0;
    for (size_t j = 0; j < search_region.size(); j++) {
      deriv_peak += (j + fit_region.start) * search_region[j];
      deriv_stddev += pow(j + fit_region.start, 2) * search_region[j];
    }
    average_line_spread = sqrt(deriv_stddev - deriv_peak * deriv_peak);

    if (vertical_edge) {
      edge_offsets->emplace_back(
          lx * deriv_peak + ly * i,
          edge[0] * deriv_peak + edge[1] * i, 1);
    } else {
      edge_offsets->emplace_back(
          lx * i + ly * deriv_peak,
          edge[0] * i + edge[1] * deriv_peak, 1);
    }
  }

  vector<pair<double, double>> line_space_edge;
  for (const auto& tmp : *edge_offsets) {
    line_space_edge.emplace_back(get<0>(tmp), get<1>(tmp));
  }
  *gp_ << "plot " << gp_->file1d(line_space_edge) << " w l\n" << endl;


  get<2>((*edge_offsets)[0]) = 0;
  get<2>(edge_offsets->back()) = 0;
  cv::imwrite("/Users/philipsalvaggio/Desktop/edge.png",
              OverlayLine(deriv, edge, *edge_offsets));

  const int kNeighborhoodSize = 2 * round(average_line_spread) + 1;
  const int kNeighborhoodHalfSize = kNeighborhoodSize / 2;
  ransac::RansacFitLine line_fitter(1);
  for (int i = 1; i < (int)edge_offsets->size() - 1; i++) {
    int start = max(0, i - kNeighborhoodHalfSize); 
    int end = min((int)edge_offsets->size(), i + kNeighborhoodHalfSize + 1);
    int num_observations = end - start + 1;

    vector<double> data(num_observations * 2, 0);
    vector<int> sample;
    double offset_mean = 0;
    for (int j = start, k = 0; j < end; j++, k++) {
      data[2*k] = get<0>((*edge_offsets)[j]);
      data[2*k+1] = get<1>((*edge_offsets)[j]);
      sample.push_back(k);

      offset_mean += data[2*k+1];
    }
    offset_mean /= num_observations;

    vector<double> model;
    line_fitter.FitLeastSquaresLine(data, sample, &model);

    double error_mean = 0, error_variance = 0;
    for (int j = start, k = 0; j < end; j++, k++) {
      double error = model[0] * get<0>((*edge_offsets)[j]) +
                     model[1] * get<1>((*edge_offsets)[j]) -
                     model[2];
      error_variance += error * error;
      error_mean += error;
    }
    error_mean /= num_observations;
    error_variance /= num_observations;
    double error_stddev = sqrt(error_variance - error_mean * error_mean);
    double how_horizontal = model[1];

    get<2>((*edge_offsets)[i]) = (1 - error_stddev) + 0.25 * how_horizontal;
  }


  /*
  const double kDerivStdDev = 0.05;
  for (size_t i = 1; i < edge_offsets->size() - 1; i++) {
    double prev_deriv =
        get<1>((*edge_offsets)[i]) - get<1>((*edge_offsets)[i-1]);
    double next_deriv = 
        get<1>((*edge_offsets)[i+1]) - get<1>((*edge_offsets)[i]);
    double mean = 0.5 * (abs(prev_deriv) + abs(next_deriv));
    double weight = exp(-(mean * mean) / (2 * kDerivStdDev * kDerivStdDev));
    get<2>((*edge_offsets)[i]) = weight;
  }
  */

  return true;
}

double JaggedEdgeMtf::FindSubpixelMax(const cv::Mat& function) {
  int size = function.size().area();

  cv::Mat_<double> obs_matrix(size, 3, CV_64FC1),
                   result_vector(size, 1, CV_64FC1);
  for (int i = 0; i < size; i++) {
    result_vector(i) = function.at<double>(i);
    obs_matrix(i, 0) = i * i;
    obs_matrix(i, 1) = i;
    obs_matrix(i, 2) = 1;
  }
  cv::Mat normal_matrix = obs_matrix.t() * obs_matrix;

  cv::Mat_<double> params =
    normal_matrix.inv() * obs_matrix.t() * result_vector;

  double deriv_slope = 2 * params(0);
  double deriv_int = params(1);
  double max_idx = -deriv_int / deriv_slope;
  return max_idx;
}

int JaggedEdgeMtf::GetSamplesPerPixel(const cv::Mat& image, const double* edge) {
  // Look at the distribution of pixels 1 pixel away from the edge.
  std::vector<float> distances;
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      float distance = edge[0]*j + edge[1]*i - edge[2];
      if (distance >= 0 && distance < 1) {
        distances.push_back(distance);
      }
    }
  }
  cv::Mat distances_mat(distances, false);

  // Create histograms of the distances to figure out what resolution can be
  // supported.
  int kMaxNumBins = 4;
  bool uniform = false;
  int num_bins = kMaxNumBins;
  float* bounds = new float[2];
  bounds[0] = 0;
  bounds[1] = 1;
  cv::Mat histogram;
  while (!uniform && num_bins > 1) {
    double uniform_expectation = ((double) distances.size()) / num_bins;
    double bin_threshold = 0.75 * uniform_expectation;
    int channel = 0;
    cv::calcHist(&distances_mat, 1, &channel, cv::Mat(), histogram, 1,
                 &num_bins, (const float**)&bounds, true, false);

    double min_bin;
    cv::minMaxIdx(histogram, &min_bin, 0);

    if (min_bin >= bin_threshold) {
      uniform = true;
    } else {
      num_bins /= 2;
    }
  }
  delete[] bounds;

  return num_bins;
}

template<typename T>
void GenerateEsfHelper(const cv::Mat& image,
                       const double* edge,
                       const JaggedEdgeMtf::offset_t& edge_offsets,
                       int samples_per_pixel,
                       std::vector<double>* esf,
                       std::vector<double>* esf_stddev,
                       std::vector<double>* bin_counts) {
  double max_distance = 60;

  double lx = -edge[1], ly = edge[0];
  if (ly < 0) {
    lx *= -1; ly *= -1;
  }

  int num_bins = samples_per_pixel * (2 * max_distance);
  double bin_size = 2 * max_distance / num_bins;
  bin_counts->resize(num_bins, 0);
  esf->resize(num_bins, 0);
  esf_stddev->resize(num_bins, 0);

  // Build up the edge spread function.
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      double line_proj = lx * j + ly * i;
      double offset, weight;
      JaggedEdgeMtf::GetOffset(line_proj, edge_offsets, offset, weight);
      double distance = edge[0]*j + edge[1]*i - offset;
      if (distance >= -max_distance && distance < max_distance) {
        int bin = (distance + max_distance) / bin_size;
        (*bin_counts)[bin] += weight;
        (*esf)[bin] += weight * image.at<T>(i, j);
        (*esf_stddev)[bin] += weight * image.at<T>(i, j) * image.at<T>(i, j);
      }
    }
  }
}

#define GENERATE_ESF_HELPER(type) \
  GenerateEsfHelper<type>(image, \
                          edge, \
                          edge_offsets, \
                          samples_per_pixel, \
                          esf, \
                          esf_stddev, \
                          &bin_counts)

void JaggedEdgeMtf::GenerateEsf(const cv::Mat& image,
                                const double* edge,
                                const offset_t& edge_offsets,
                                int samples_per_pixel,
                                std::vector<double>* esf,
                                std::vector<double>* esf_stddev) {
  if (!esf) return;

  std::vector<double> bin_counts;
  switch (image.depth()) {
    case CV_8U: GENERATE_ESF_HELPER(uint8_t); break;
    case CV_8S: GENERATE_ESF_HELPER(int8_t); break;
    case CV_16U: GENERATE_ESF_HELPER(uint16_t); break;
    case CV_16S: GENERATE_ESF_HELPER(int16_t); break;
    case CV_32S: GENERATE_ESF_HELPER(int32_t); break;
    case CV_32F: GENERATE_ESF_HELPER(float); break;
    case CV_64F: GENERATE_ESF_HELPER(double); break;
  }

  for (size_t i = 0; i < esf->size(); i++) {
    (*esf)[i] /= bin_counts[i];
    (*esf_stddev)[i] /= bin_counts[i];
    (*esf_stddev)[i] = sqrt(max(0., (*esf_stddev)[i] - (*esf)[i] * (*esf)[i]));
  }
}

void JaggedEdgeMtf::SmoothEsf(std::vector<double>* esf) {
  for (size_t i = 0; i < esf->size(); i++) {
    size_t prev = std::max(0, (int)i - 1);
    size_t next = std::min(i + 1, esf->size() - 1);

    (*esf)[i] = ((*esf)[i] + (*esf)[next] + (*esf)[prev]) / 3.0;
  }
}

void JaggedEdgeMtf::PlotEsf(const std::vector<double>& esf,
                            const std::vector<double>& esf_stddev) {
  vector<pair<double, double>> esf_data;
  for (size_t i = 0; i < esf.size(); i++) {
    esf_data.emplace_back(esf[i], esf_stddev[i]);
  }

  *gp_ << "unset key\n"
       << "set xlabel \"Pixels\"\n"
       << "set ylabel \"ESF\"\n"
       << "plot " << gp_->file1d(esf_data) << " w errorbars\n"
       << endl;
}


void JaggedEdgeMtf::GetOffset(double edge_coord,
                              const offset_t& offsets,
                              double& offset,
                              double& weight) {
  auto lower = lower_bound(begin(offsets), end(offsets), edge_coord,
      [] (const tuple<double, double, double>& pt, double query) {
        return get<0>(pt) < query;
      });

  if (lower == begin(offsets)) {
    offset = get<1>(offsets[0]);
    weight = get<2>(offsets[0]);
  } else if (lower == end(offsets)) {
    offset = get<1>(offsets.back());
    weight = get<2>(offsets.back());
  } else {
    int gt_index = lower - begin(offsets);
    int lt_index = gt_index - 1;

    double range = get<0>(offsets[gt_index]) - get<0>(offsets[lt_index]);
    double blend = (edge_coord - get<0>(offsets[lt_index])) / range;

    offset = (1 - blend) * get<1>(offsets[lt_index]) +
             blend * get<1>(offsets[gt_index]);
    weight = (1 - blend) * get<2>(offsets[lt_index]) +
             blend * get<2>(offsets[gt_index]);
  }
}
