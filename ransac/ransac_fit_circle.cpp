// File Description
// Author: Philip Salvaggio

#include "ransac_fit_circle.h"

#include <cmath>
#include <iostream>

RansacFitOriginCircle::RansacFitOriginCircle(double threshold)
    : threshold_(threshold) {}

RansacFitOriginCircle::~RansacFitOriginCircle() {}

bool RansacFitOriginCircle::RansacDegeneracyScreen(
    const data_t& data,
    const std::vector<int>& random_sample) const {
  (void)data;
  (void)random_sample;
  return false;
}

void RansacFitOriginCircle::RansacFitModel(
    const data_t& data,
    const std::vector<int>& random_sample,
    std::vector<model_t*>* models) const {
  double x = data[2*random_sample[0]];
  double y = data[2*random_sample[0]+1];

  models->push_back(new std::vector<double>(1, sqrt(x*x + y*y)));
}

int RansacFitOriginCircle::RansacGetInliers(
    const data_t& data,
    const std::vector<model_t*>& models,
    std::list<int>* inliers) const {

  double model_r2 = (*(models[0]))[0] * (*(models[0]))[0];
  double sq_threshold = threshold_ * threshold_;
  for (size_t i = 0; i < data.size() / 2; i++) {
    double r2 = data[2*i]*data[2*i] + data[2*i+1]*data[2*i+1];
    if (fabs(r2 - model_r2) <= sq_threshold) {
      inliers->push_back(i);
    }
  }

  return 0;
}
