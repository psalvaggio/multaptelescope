// A class to interpolate a 2D MTF from sample points.
// Author: Philip Salvaggio

#ifndef MTF_INTERPOLATOR_H
#define MTF_INTERPOLATOR_H

#include <vector>
#include <opencv/cv.h>

class MtfInterpolator {
 public:
  // Constructor
  MtfInterpolator();

  // Destructor
  virtual ~MtfInterpolator();

  // Get the interpolated MTF.
  //
  // Arguments:
  //  samples            A list of the data. Should be in the form
  //                     [xi1, eta1, MTF(xi1, eta1), ...,
  //                      xin, etan, MTF(xin, etan)]
  //  num_samples        The number of samples.
  //  rows               Number of rows in the target MTF.
  //  cols               Number of columns in the target MTF.
  //  circular_symmetry  Whether to assume the MTF is circularly symmetric.
  //  mtf                Output: Pointer to output array. Will be overwritten.
  void GetMtf(double* samples,
              int num_samples,
              int rows,
              int cols,
              bool circular_symmetry,
              cv::Mat* mtf);
};

#endif  // MTF_INTERPOLATOR_H
