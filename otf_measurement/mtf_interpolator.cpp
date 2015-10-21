// File Description
// Author: Philip Salvaggio

#include "mtf_interpolator.h"

#include "base/assertions.h"

#include <iostream>
#include <list>

using namespace std;
using namespace cv;

MtfInterpolator::MtfInterpolator() {}

MtfInterpolator::~MtfInterpolator() {}

void MtfInterpolator::GetMtf(const std::vector<MTF>& profiles,
                             const std::vector<double>& angles,
                             int rows,
                             int cols,
                             int reflections,
                             cv::Mat* output_mtf) {
  if (!output_mtf) return;

  CHECK(profiles.size() == angles.size());

  const double kAngleUpper = 2 * M_PI / reflections;

  output_mtf->create(rows, cols, CV_64FC1);
  for (int y = 0; y < rows; y++) {
    for (int x = 0; x < cols; x++) {
      // Compute the radius, ranges from 0 - 0.5 on the shorter axis.
      double r = sqrt(pow(rows / 2. - y, 2) +
                      pow(cols / 2. - x, 2));
      r /= min(rows, cols);

      // Compute the angle in the range [0, kAngleUpper)
      double theta = atan2(rows / 2. - y, cols / 2. - x);
      while (theta < 0) theta += 2 * M_PI;
      while (theta > kAngleUpper) theta -= kAngleUpper;

      // Find the two profiles around this point.
      auto lower = lower_bound(begin(angles), end(angles), theta);
      int ang_gt_index = lower - begin(angles);
      int ang_lt_index = ang_gt_index - 1;

      // Come up with the linear interpolation weights for the angular
      // dimension.
      vector<pair<int, double>> interp_weights;

      // If the angle is greater than our last profile, interpolate between the
      // last and the first.
      if (ang_gt_index == int(angles.size())) {
        double diff = theta - angles[ang_lt_index];
        double blend = diff / ((angles[0] + kAngleUpper) - angles.back());
        interp_weights.emplace_back(angles.size() - 1, 1 - blend);
        interp_weights.emplace_back(0, blend);
    
      // If the angle is less than our first profile, interpolate between the
      // last and the first.
      } else if (ang_gt_index == 0) {
        double diff = theta - (angles.back() - kAngleUpper);
        double blend = diff / (angles[0] - (angles.back() - kAngleUpper));
        interp_weights.emplace_back(angles.size() - 1, 1 - blend);
        interp_weights.emplace_back(0, blend);

      } else {
        double diff = theta - angles[ang_lt_index];
        double blend = diff / (angles[ang_gt_index] - angles[ang_lt_index]);
        interp_weights.emplace_back(ang_lt_index, 1 - blend);
        interp_weights.emplace_back(ang_gt_index, blend);
      }

      // Perform the bilinear interpolation in polar frequency space.
      double mtf_val = 0;
      for (const auto& angle_weight : interp_weights) {
        const MTF& profile = profiles[angle_weight.first];

        // Find the two bounding MTF samples.
        auto r_lower = lower_bound(begin(profile[0]), end(profile[0]), r);
        int r_gt_index = r_lower - begin(profile[0]);
        int r_lt_index = r_gt_index - 1;

        // If we're beyond the smaples, assume 0.
        if (r_gt_index == int(profile[0].size())) {
          mtf_val += 0;
        } else if (r_lt_index < 0) {  // Just here for -0s.
          mtf_val += angle_weight.second;
        } else {
          double blend = (r - profile[0][r_lt_index]) /
                         (profile[0][r_gt_index] - profile[0][r_lt_index]);
          mtf_val += (profile[1][r_lt_index] * (1 - blend) +
                      profile[1][r_gt_index] * blend) * angle_weight.second;
        }
      }

      output_mtf->at<double>(y, x) = mtf_val;
    }
  }
}
