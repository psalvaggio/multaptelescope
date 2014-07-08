// File Description
// Author: Philip Salvaggio

#ifndef ABERRATION_FACTORY_H
#define ABERRATION_FACTORY_H

#include <opencv/cv.h>
#include <vector>

#include "base/macros.h"

class AberrationFactory {
 public:
  // Generate an aberration WFE array using Zernike polynomials.
  //
  // Arguments:
  //  weights  A list of relative weights for each Zernike polynomials.
  //           Up to 35 are supported, but are assumed to be 0 is not specfied.
  //           A full listing can be found at
  //           http://wyant.optics.arizona.edu/zernikes/zernikes.htm
  //           For modeling up to 4th order wavefront aberrations:
  //           0 - Piston
  //           1 - Tilt X
  //           2 - Tilt Y
  //           3 - Defocus
  //           4 - X Astigmatism
  //           5 - Y Astigmatism
  //           6 - Coma X
  //           7 - Coma Y
  //           8 - Spherical
  //  output  Output: the wavefront error map.
  static void ZernikeAberrations(const std::vector<double>& weights,
                                 size_t output_size,
                                 cv::Mat* output);


 private:
  NO_CONSTRUCTION(AberrationFactory);
};

#endif  // ABERRATION_FACTORY_H
