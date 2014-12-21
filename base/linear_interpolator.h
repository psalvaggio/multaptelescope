// A class that represents samples of a 1D function. Intermediate values can be
// obtained through linear interpolation.
// Author: Philip Salvaggio

#ifndef LINEAR_INTERPOLATOR_H
#define LINEAR_INTERPOLATOR_H

#include <vector>

namespace mats {

class LinearInterpolator {
 public:
  // Perform linear interpolation.
  //
  // Arguments:
  //  independent_samples  Independent variable values of function samples.
  //                       These MUST be sorted in ascending order.
  //  dependent_samples    Corresponding dependent variable values of samples.
  //  independent_queries  Independent variable values at which the function
  //                       should be evaluatated.
  //  dependent_values     Output: Results of interpolation.
  static void Interpolate(const std::vector<double>& independent_samples,
                          const std::vector<double>& dependent_samples,
                          const std::vector<double>& independent_queries,
                          std::vector<double>* dependent_values);
};

}

#endif  // LINEAR_INTERPOLATOR_H
