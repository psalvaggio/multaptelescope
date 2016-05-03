// File Description
// Author: Philip Salvaggio

#ifndef RANSAC_FIT_LINE_H
#define RANSAC_FIT_LINE_H

#include "ransac.h"

namespace ransac {

class RansacFitLine : public RansacImpl<std::vector<double>,
                                        std::vector<double>> {
 public:
  explicit RansacFitLine(double threshold);

  double threshold() const { return threshold_; }
  void set_threshold(double threshold) { threshold_ = threshold; }

  void FitModel(const data_t& data,
                const std::vector<int>& random_sample,
                std::vector<model_t>* models) const override;

  int GetInliers(const data_t& data,
                 const std::vector<model_t>& models,
                 std::vector<int>* inliers) const override;

  void FitLeastSquaresLine(const data_t& data,
                           const std::vector<int>& sample,
                           model_t* model) const;
 private:
  double threshold_;
};

}

#endif  // RANSAC_FIT_LINE_H
