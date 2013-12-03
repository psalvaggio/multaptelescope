// File Description
// Author: Philip Salvaggio

#include "slant_edge_mtf.h"

#include "base/opencv_utils.h"
#include "otf_measurement/hamming_window.h"
#include "ransac/ransac.h"
#include "ransac/ransac_fit_line.h"

#include <opencv/highgui.h>
#include <cmath>

#include <fftw3.h>

SlantEdgeMtf::SlantEdgeMtf() {}

SlantEdgeMtf::~SlantEdgeMtf() {}

void SlantEdgeMtf::Analyze(const cv::Mat& image,
                           double* orientation,
                           std::vector<double>* mtf) {
  if (!orientation || !mtf) return;

  // Detect the edge in the image.
  double edge[3];
  bool found_edge = DetectEdge(image, edge);
  *orientation = atan2(-edge[0], edge[1]);
  if (*orientation >= M_PI / 2) *orientation -= M_PI;
  if (*orientation < -M_PI / 2) *orientation += M_PI;

  imshow("Detected Edge", OverlayLine(image, edge));
  //std::cerr << "Detected Orientation " << *orientation * 180 / M_PI
            //<< std::endl;


  int num_bins = GetSamplesPerPixel(image, edge);
  //std::cerr << "Using " << num_bins << " samples per pixel." << std::endl;

  // Compute and smooth the edge spread function (ESF)
  std::vector<double> esf;
  GenerateEsf(image, edge, num_bins, &esf);
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
    lsf[i][0] *= hamming[i];
  }
  delete[] hamming;

  fftw_plan fft_plan  = fftw_plan_dft_1d(esf.size(), lsf, otf,
                                         FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(fft_plan);

  fftw_destroy_plan(fft_plan);
  fftw_free(lsf);


  int nyquist_idx = esf.size() / (2*num_bins);

  double peak_mtf = sqrt(otf[0][0]*otf[0][0] + otf[0][1]*otf[0][1]);
  mtf->push_back(1);
  for (size_t i = 1; i <= nyquist_idx; i++) {
    double positive_mag = sqrt(otf[i][0]*otf[i][0] + otf[i][1]*otf[i][1]);
    double negative_mag = sqrt(otf[esf.size()-i][0] * otf[esf.size()-i][0] +
                               otf[esf.size()-i][1] * otf[esf.size()-i][1]);
    mtf->push_back(0.5*(positive_mag + negative_mag) / peak_mtf);
  }

  fftw_free(otf);
}

cv::Mat SlantEdgeMtf::OverlayLine(const cv::Mat& image, const double* line) {
  // Replicate into RGB planes.
  std::vector<cv::Mat> rgb_planes;
  for (int i = 0; i < 3; i++) {
    rgb_planes.push_back(cv::Mat());
    image.copyTo(rgb_planes[i]);
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

  cv::Mat rgb;
  cv::merge(rgb_planes, rgb);
  return rgb;
}


bool SlantEdgeMtf::DetectEdge(const cv::Mat& image, double* edge) {
  if (!edge) return false;

  // Take the x- and y-derivatives of the image.
  cv::Mat dy, dx;
  cv::absdiff(image,
              circshift(image, cv::Point2f(-1, 0), cv::BORDER_REFLECT), dx);
  cv::absdiff(image,
              circshift(image, cv::Point2f(0, -1), cv::BORDER_REFLECT), dy);
  dy.convertTo(dy, CV_64FC1);
  dx.convertTo(dx, CV_64FC1);

  // Get an edge-magnitude map of the image.
  cv::Mat edges = dy.mul(dy) + dx.mul(dx);

  // If the edge is vertical, there should be more energy in the x-derivative.
  // Take the sum of entire derivative image to determine which orientation the
  // edge is in.
  double total_dx = (sum(dx))[0];
  double total_dy = (sum(dy))[0];

  // The maximum value of the derivative should occur at the center of the
  // line. Scan for the max value and create a set of input points for RANSAC.
  std::vector<double> line_pts;
  cv::Mat test = cv::Mat::zeros(image.rows, image.cols, CV_64FC1);
  if (total_dy > total_dx) {
    int maxLoc[2];
    for (int i = 0; i < image.cols; i++) {
      cv::Mat col = edges.col(i);
      minMaxIdx(col, 0, 0, 0, maxLoc);
      line_pts.push_back(i);
      line_pts.push_back(maxLoc[0]);
    }
  } else {
    int maxLoc[2];
    for (int i = 0; i < image.rows; i++) {
      cv::Mat row = edges.row(i);
      minMaxIdx(row, 0, 0, 0, maxLoc);
      line_pts.push_back(maxLoc[1]);
      line_pts.push_back(i);
    }
  }

  //imshow("line points", line_points);
  /*
  test = ByteScale(test);
  std::vector<cv::Mat> test_planes;
  for (int i = 0; i < 3; i++) {
    test_planes.push_back(cv::Mat());
    test.copyTo(test_planes[i]);
  }
  for (int i = 0; i < line_pts.size(); i+=2){ 
    if (line_pts[i] >= 0 && line_pts[i+1] >= 0 && line_pts[i] <= image.cols - 1
        && line_pts[i+1] <= image.rows - 1) {
      test_planes[0].at<uint8_t>(line_pts[i+1], line_pts[i]) = 0;
      test_planes[1].at<uint8_t>(line_pts[i+1], line_pts[i]) = 255;
      test_planes[2].at<uint8_t>(line_pts[i+1], line_pts[i]) = 0;
    }
  }
  cv::Mat test_rgb;
  cv::merge(test_planes, test_rgb);
  imshow("test", test_rgb);
  */


  // Use RANSAC to find the inlier points on the line.
  RansacFitLine line_fitter(3);
  RansacFitLine::model_t* best_model = NULL;
  std::list<int> inliers;
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
    std::cerr << "RANSAC Error: " << RansacErrorString(error) << std::endl;
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

int SlantEdgeMtf::GetSamplesPerPixel(const cv::Mat& image, const double* edge) {
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

void SlantEdgeMtf::GenerateEsf(const cv::Mat& image,
                               const double* edge,
                               int samples_per_pixel,
                               std::vector<double>* esf) {
  if (!esf) return;

  double max_distance = 20;

  int num_bins = samples_per_pixel * (2 * max_distance);
  double bin_size = 2 * max_distance / num_bins;
  std::vector<int> bin_counts(num_bins, 0);
  esf->resize(num_bins, 0);

  // Build up the edge spread function.
  uint8_t* image_data = (uint8_t*)image.data;
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      double distance = edge[0]*j + edge[1]*i - edge[2];
      if (distance >= -max_distance && distance < max_distance) {
        int bin = (distance + max_distance) / bin_size;
        bin_counts[bin]++;
        (*esf)[bin] += image_data[i*image.cols + j];
      }
    }
  }

  for (size_t i = 0; i < esf->size(); i++) {
    (*esf)[i] /= bin_counts[i];
  }
}

void SlantEdgeMtf::SmoothEsf(std::vector<double>* esf) {
  for (size_t i = 0; i < esf->size(); i++) {
    size_t prev = std::max(0, (int)i - 1);
    size_t next = std::min(i + 1, esf->size() - 1);

    (*esf)[i] = ((*esf)[i] + (*esf)[next] + (*esf)[prev]) / 3.0;
  }
}
