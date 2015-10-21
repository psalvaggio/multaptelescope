// A class to interpolate a 2D MTF from sample points.
// Author: Philip Salvaggio

#ifndef MTF_INTERPOLATOR_H
#define MTF_INTERPOLATOR_H

#include <vector>
#include <opencv2/core/core.hpp>

class MtfInterpolator {
 public:  // Types
  using MTF = std::vector<std::vector<double>>;  // [[frequencies], [MTF vals]]

 public:
  // Constructor
  MtfInterpolator();

  // Destructor
  virtual ~MtfInterpolator();

  // Get the interpolated MTF.
  //
  // Arguments:
  //  profiles           A list of the MTF profiles.
  //  angles             A list of angles that correspond to profiles.
  //                     NOTE: Must be in ascending order.
  //  rows               Number of rows in the target MTF.
  //  cols               Number of columns in the target MTF.
  //  reflections        The number of times to reflect the profiles to fill the
  //                     entire MTF.
  //  output_mtf         Output: Pointer to output array. Will be overwritten.
  void GetMtf(const std::vector<MTF>& profiles,
              const std::vector<double>& angles,
              int rows,
              int cols,
              int reflections,
              cv::Mat* output_mtf);
};

#endif  // MTF_INTERPOLATOR_H
