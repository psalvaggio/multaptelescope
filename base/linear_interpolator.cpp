// File Description
// Author: Philip Salvaggio

#include "linear_interpolator.h"

#include <algorithm>

using namespace std;

namespace mats {

void LinearInterpolator::Interpolate(const vector<double>& independent_samples,
                                     const vector<double>& dependent_samples,
                                     const vector<double>& independent_queries,
                                     vector<double>* dependent_values) {
  if (!dependent_values) return;
  dependent_values->clear();

  for (double query : independent_queries) {
    auto lower = lower_bound(begin(independent_samples),
                             end(independent_samples),
                             query);

    if (lower == begin(independent_samples)) {
      dependent_values->push_back(dependent_samples.front());
    } else if (lower == end(independent_samples)) {
      dependent_values->push_back(dependent_samples.back());
    } else {
      int gt_index = lower - begin(independent_samples);
      int lt_index = gt_index - 1;
      double range =
          independent_samples[gt_index] - independent_samples[lt_index];

      double blend = (query - independent_samples[lt_index]) / range;

      dependent_values->push_back((1 - blend) * dependent_samples[lt_index] +
                                  blend * dependent_samples[gt_index]);
    }
  }
}

}
