// File Description
// Author: Philip Salvaggio

#include "slant_edge_mtf.h"

#include "base/opencv_utils.h"
#include "otf_measurement/hamming_window.h"
#include "ransac/ransac.h"
#include "ransac/ransac_fit_line.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>

#include <fftw3.h>

using namespace std;
using namespace cv;

SlantEdgeMtf::SlantEdgeMtf() {
  gp_ = new Gnuplot();
  local_gp_ = true;
}

SlantEdgeMtf::~SlantEdgeMtf() {
  if (local_gp_ && gp_) {
    *gp_ << endl;
    delete gp_;
  }
}

void SlantEdgeMtf::Analyze(const Mat& image,
                           double* orientation,
                           vector<double>* mtf) {
  if (!orientation || !mtf) return;

  // Detect the edge in the image.
  double edge[3];
  DetectEdge(image, edge);
  *orientation = atan2(-edge[0], edge[1]);
  if (*orientation >= M_PI / 2) *orientation -= M_PI;
  if (*orientation < -M_PI / 2) *orientation += M_PI;

  //imshow("Detected Edge", OverlayLine(image, edge));

  int num_bins = GetSamplesPerPixel(image, edge);

  // Compute and smooth the edge spread function (ESF)
  vector<double> esf, esf_stddevs;
  GenerateEsf(image, edge, num_bins, &esf, &esf_stddevs);
  SmoothEsf(&esf);
  //PlotEsf(esf/*, esf_stddevs*/);

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
    double blur_neg_mag = sqrt(pow(blur_otf[esf.size()-i][0], 2) +
                               pow(blur_otf[esf.size()-i][1], 2));
    double blur_mag = 0.5 * (blur_pos_mag + blur_neg_mag) / peak_blur_mtf;
    mtf->push_back(0.5*(positive_mag + negative_mag) / peak_mtf / blur_mag);
  }

  fftw_free(otf);
  fftw_free(blur_otf);
}

Mat SlantEdgeMtf::OverlayLine(const Mat& image, const double* line) {
  Mat byte_scaled_image;
  ByteScale(image, byte_scaled_image);

  // Replicate into RGB planes.
  vector<Mat> rgb_planes;
  for (int i = 0; i < 3; i++) {
    rgb_planes.push_back(Mat());
    byte_scaled_image.copyTo(rgb_planes[i]);
  }
  uint8_t* b = rgb_planes[0].data;
  uint8_t* g = rgb_planes[1].data;
  uint8_t* r = rgb_planes[2].data;

  // Draw the line.
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      double distance = line[0]*j + line[1]*i - line[2];
      if (fabs(distance) < 0.5) {
        r[i*image.cols + j] = 0;
        g[i*image.cols + j] = 255;
        b[i*image.cols + j] = 0;
      }
    }
  }

  Mat rgb;
  merge(rgb_planes, rgb);
  return rgb;
}

void SlantEdgeMtf::SetGnuplot(Gnuplot* gp) {
  if (gp_ == gp) return;

  if (local_gp_ && gp_) {
    delete gp_;
  }

  gp_ = gp;
  local_gp_ = false;
}

bool SlantEdgeMtf::DetectEdge(const Mat& image, double* edge) {
  if (!edge) return false;

  // Take the x- and y-derivatives of the image.
  Mat dy, dx;
  absdiff(image, circshift(image, Point2f(-1, 0), BORDER_REFLECT), dx);
  absdiff(image, circshift(image, Point2f(0, -1), BORDER_REFLECT), dy);
  dy.convertTo(dy, CV_64FC1);
  dx.convertTo(dx, CV_64FC1);

  // Get an edge-magnitude map of the image.
  Mat edges = dy.mul(dy) + dx.mul(dx);

  // If the edge is vertical, there should be more energy in the x-derivative.
  // Take the sum of entire derivative image to determine which orientation the
  // edge is in.
  double total_dx = (sum(dx))[0];
  double total_dy = (sum(dy))[0];

  // The maximum value of the derivative should occur at the center of the
  // line. Scan for the max value and create a set of input points for RANSAC.
  vector<double> line_pts;
  Mat test = Mat::zeros(image.rows, image.cols, CV_64FC1);
  if (total_dy > total_dx) {
    int maxLoc[2];
    for (int i = 0; i < image.cols; i++) {
      Mat col = edges.col(i);
      minMaxIdx(col, 0, 0, 0, maxLoc);
      line_pts.push_back(i);
      line_pts.push_back(maxLoc[0]);
    }
  } else {
    int maxLoc[2];
    for (int i = 0; i < image.rows; i++) {
      Mat row = edges.row(i);
      minMaxIdx(row, 0, 0, 0, maxLoc);
      line_pts.push_back(maxLoc[1]);
      line_pts.push_back(i);
    }
  }

  // Use RANSAC to find the inlier points on the line.
  RansacFitLine line_fitter(3);
  RansacFitLine::model_t* best_model = NULL;
  list<int> inliers;
  ransac::Error_t error = ransac::Ransac(
      line_fitter,
      line_pts,
      line_pts.size() / 2,
      2,
      10,
      10000,
      &best_model,
      &inliers);

  // Bail out if RANSAC failed.
  if (error != ransac::RansacSuccess) {
    cerr << "RANSAC Error: " << RansacErrorString(error) << endl;
    return false;
  }

  // Use least-squares on the inliers to fit a more stable line.
  delete best_model;
  line_fitter.FitLeastSquaresLine(line_pts, inliers, &best_model);

  edge[0] = (*best_model)[0];
  edge[1] = (*best_model)[1];
  edge[2] = (*best_model)[2];

  delete best_model;

  return true;
}

int SlantEdgeMtf::GetSamplesPerPixel(const Mat& image, const double* edge) {
  // Look at the distribution of pixels 1 pixel away from the edge.
  vector<float> distances;
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      float distance = edge[0]*j + edge[1]*i - edge[2];
      if (distance >= 0 && distance < 1) {
        distances.push_back(distance);
      }
    }
  }
  Mat distances_mat(distances, false);

  // Create histograms of the distances to figure out what resolution can be
  // supported.
  int kMaxNumBins = 4;
  bool uniform = false;
  int num_bins = kMaxNumBins;
  float* bounds = new float[2];
  bounds[0] = 0;
  bounds[1] = 1;
  Mat histogram;
  while (!uniform && num_bins > 1) {
    double uniform_expectation = ((double) distances.size()) / num_bins;
    double bin_threshold = 0.75 * uniform_expectation;
    int channel = 0;
    calcHist(&distances_mat, 1, &channel, Mat(), histogram, 1,
             &num_bins, (const float**)&bounds, true, false);

    double min_bin;
    minMaxIdx(histogram, &min_bin, 0);

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
void GenerateEsfHelper(const Mat& image,
                       const double* edge,
                       int samples_per_pixel,
                       vector<double>* esf,
                       vector<double>* esf_stddevs,
                       vector<int>* bin_counts) {
  double max_distance = 60;

  int num_bins = samples_per_pixel * (2 * max_distance);
  double bin_size = 2 * max_distance / num_bins;
  bin_counts->resize(num_bins, 0);
  esf->resize(num_bins, 0);
  esf_stddevs->resize(num_bins, 0);

  // Build up the edge spread function.
  //T* image_data = (T*)image.data;
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      double distance = edge[0]*j + edge[1]*i - edge[2];
      if (distance >= -max_distance && distance < max_distance) {
        int bin = (distance + max_distance) / bin_size;
        (*bin_counts)[bin]++;
        (*esf)[bin] += image.at<T>(i, j);
        (*esf_stddevs)[bin] += pow(image.at<T>(i, j), 2);
      }
    }
  }
}

#define GENERATE_ESF_HELPER(type) \
  GenerateEsfHelper<type>(image, edge, samples_per_pixel, \
                          esf, esf_stddevs, &bin_counts)

void SlantEdgeMtf::GenerateEsf(const Mat& image,
                               const double* edge,
                               int samples_per_pixel,
                               vector<double>* esf,
                               vector<double>* esf_stddevs) {
  if (!esf) return;

  vector<int> bin_counts;
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
    if (bin_counts[i] > 0) { 
      (*esf)[i] /= bin_counts[i];
      (*esf_stddevs)[i] /= bin_counts[i];
      (*esf_stddevs)[i] = sqrt((*esf_stddevs)[i] - pow((*esf)[i], 2));
    } else {
      (*esf)[i] = -1;
    }
  }
}

void SlantEdgeMtf::SmoothEsf(vector<double>* esf) {
  for (size_t i = 0; i < esf->size(); i++) {
    size_t prev = max(0, (int)i - 1);
    size_t next = min(i + 1, esf->size() - 1);

    double val = 0, norm = 0;
    for (size_t j = prev; j <= next; j++) {
      if ((*esf)[j] >= 0) {
        val += (*esf)[j];
        norm++;
      }
    }

    if (norm > 0) {
      (*esf)[i] = val / norm;
    } else {
      for (size_t offset = 2; offset < esf->size(); offset++) {
        int lt_index = i - offset,
            gt_index = i + offset;
        if (lt_index > 0 && (*esf)[lt_index] > 0) {
          (*esf)[i] = (*esf)[lt_index];
          break;
        } else if (gt_index < int(esf->size()) && (*esf)[gt_index] > 0) {
          (*esf)[i] = (*esf)[gt_index];
          break;
        }
      }
    }
  }
}

void SlantEdgeMtf::PlotEsf(const vector<double>& esf) {
  *gp_ << "set xlabel \"Pixels\"\n"
      << "set ylabel \"ESF\"\n"
      << "plot " << gp_->file1d(esf) << " w l\n"
      << endl;
}

void SlantEdgeMtf::PlotEsf(const vector<double>& esf,
                           const vector<double>& esf_stddevs) {
  vector<pair<double, double>> esf_data;
  for (size_t i = 0; i < esf.size(); i++) {
    esf_data.emplace_back(esf[i], esf_stddevs[i]);
  }

  *gp_ << "unset key\n"
       << "set xlabel \"Pixels\"\n"
       << "set ylabel \"ESF\"\n"
       << "plot " << gp_->file1d(esf_data) << " w errorbars\n"
       << endl;
}
