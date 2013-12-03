// File Description
// Author: Philip Salvaggio

#ifndef RANSAC_FIT_LINE_H
#define RANSAC_FIT_LINE_H

#include <list>
#include <vector>

class RansacFitLine {
 public:
  typedef std::vector<double> data_t;
  typedef std::vector<double> model_t;

  RansacFitLine(double threshold);
  ~RansacFitLine();

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

  void FitLeastSquaresLine(const data_t& data,
                           const std::list<int>& sample,
                           model_t** model) const;
 private:
  double threshold_;
};

#endif  // RANSAC_FIT_LINE_H
