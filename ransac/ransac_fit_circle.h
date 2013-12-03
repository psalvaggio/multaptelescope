// File Description
// Author: Philip Salvaggio

#ifndef RANSAC_FIT_CIRCLE_H
#define RANSAC_FIT_CIRCLE_H

#include <list>
#include <vector>

class RansacFitOriginCircle {
 public:
  typedef std::vector<double> data_t;
  typedef std::vector<double> model_t;

  RansacFitOriginCircle(double threshold);
  ~RansacFitOriginCircle();

  double threshold() const { return threshold_; }
  void set_threshold(double threshold) { threshold_ = threshold; }

  bool RansacDegeneracyScreen(const data_t& data,
                              const std::vector<int>& random_sample) const;

  void RansacFitModel(const data_t& data,
                      const std::vector<int>& random_sample,
                      std::vector<model_t*>* models) const;

  int RansacGetInliers(const data_t& data,
                       const std::vector<model_t*>& models,
                       std::list<int>* inliers) const;

 private:
  double threshold_;
};

#endif  // RANSAC_FIT_CIRCLE_H
