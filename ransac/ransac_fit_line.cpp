// File Description
// Author: Philip Salvaggio

#include "ransac_fit_line.h"

#include <cmath>
#include <opencv2/core/core.hpp>

RansacFitLine::RansacFitLine(double threshold) : threshold_(threshold) {}

RansacFitLine::~RansacFitLine() {}

bool RansacFitLine::RansacDegeneracyScreen(
    const data_t& data,
    const std::vector<int>& random_sample) const {
  (void)data;
  (void)random_sample;
  return false;
}

void RansacFitLine::RansacFitModel(const data_t& data,
                                   const std::vector<int>& random_sample,
                                   std::vector<model_t*>* models) const {
  double x1 = data[2*random_sample[0]];
  double y1 = data[2*random_sample[0]+1];
  double x2 = data[2*random_sample[1]];
  double y2 = data[2*random_sample[1]+1];

  std::vector<double>* model = new std::vector<double>();

  double dx = x2 - x1;
  double dy = y2 - y1;
  double slope_mag = sqrt(dx*dx + dy*dy);
  model->push_back(-dy / slope_mag);
  model->push_back(dx / slope_mag);
  model->push_back(model->at(0)*x1 + model->at(1)*y1);

  models->push_back(model);
}

int RansacFitLine::RansacGetInliers(const data_t& data,
                                    const std::vector<model_t*>& models,
                                    std::list<int>* inliers) const {
  for (size_t i = 0; i < data.size() / 2; i++) {
    double distance = fabs(data[2*i] * models[0]->at(0) +
                           data[2*i+1] * models[0]->at(1) -
                           models[0]->at(2));
    if (distance <= threshold_) {
      inliers->push_back(i);
    }
  }
  
  return 0;
}

void RansacFitLine::FitLeastSquaresLine(const data_t& data,
                                        const std::list<int>& sample,
                                        model_t** model) const {
  // [x1 y1 -1]         [0]
  // [x2 y2 -1]   [a]   [0]
  // [  ...   ] * [b] = [0]
  // [xn yn -1]   [c]   [0]
  cv::Mat obs_matrix(sample.size(), 2, CV_64FC1);
  double* obs_data = (double*)obs_matrix.data;

  std::list<int>::const_iterator it = sample.begin();
  double mean_x = 0, mean_y = 0;
  for (int i = 0; it != sample.end(); it++, i++) {
    obs_data[2*i + 0] = data[2*(*it) + 0];
    obs_data[2*i + 1] = data[2*(*it) + 1];
    mean_x += obs_data[2*i + 0];
    mean_y += obs_data[2*i + 1];
  }
  mean_x /= sample.size();
  mean_y /= sample.size();

  for (size_t i = 0; i < sample.size(); i++) {
    obs_data[2*i + 0] -= mean_x;
    obs_data[2*i + 1] -= mean_y;
  }

  cv::Mat normal_matrix = obs_matrix.t() * obs_matrix;

  cv::Mat U, S, Vt;
  cv::SVD::compute(normal_matrix, S, U, Vt, cv::SVD::FULL_UV);

  *model = new std::vector<double>();

  (*model)->push_back(Vt.at<double>(1, 0));
  (*model)->push_back(Vt.at<double>(1, 1));
  (*model)->push_back(Vt.at<double>(1, 0) * mean_x +
                      Vt.at<double>(1, 1) * mean_y);
}
