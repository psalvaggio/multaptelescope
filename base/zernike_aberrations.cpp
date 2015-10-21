// File Description
// Author: Philip Salvaggio

#include "zernike_aberrations.h"

#include <cmath>
#include <iostream>

using namespace cv;
using std::vector;

ZernikeAberrations::ZernikeAberrations() {}

ZernikeAberrations::~ZernikeAberrations() {}

void ZernikeAberrations::aberrations(const vector<double>& weights,
                                     size_t output_size,
                                     Mat* output) {
  if (!output) return;

  const int kSize = output_size;
  const double kCenter = 0.5 * (output_size - 1);

  if (output->rows != kSize || output->cols != kSize) {
    output->create(output_size, output_size, CV_64F);
  }
  *output = Scalar(0);

  for (int i = 0; i < kSize; i++) {
    double y = i - kCenter;
    for (int j = 0; j < kSize; j++) {
      double x = j - kCenter;
      double rho = sqrt(x*x + y*y) / kCenter;

      if (rho > 1) {
        continue;
      }

      double theta = atan2(y, x);
      double cos_theta = cos(theta);
      double sin_theta = sin(theta);

      double wfe = 0;
      for (size_t j = 0; j < weights.size(); j++) {
        switch(j) {
          case 0:
            wfe += weights[0]; break;
          case 1:
            wfe += weights[1] * rho * cos_theta; break;
          case 2:
            wfe += weights[2] * rho * sin_theta; break;
          case 3:
            wfe += weights[3] * (rho*rho - 1); break;
          case 4:
            wfe += weights[4] * rho * rho * cos(2 * theta); break;
          case 5:
            wfe += weights[5] * rho * rho * sin(2 * theta); break;
          case 6:
            wfe += weights[6] * rho * (3 * rho * rho - 2) * cos_theta; break;
          case 7:
            wfe += weights[7] * rho * (3 * rho * rho - 2) * sin_theta; break;
          case 8:
            wfe += weights[8] * (1 - 6 * rho * rho + 6 * pow(rho, 4)); break;
        }
      }
      output->at<double>(i, j) = wfe;
    }
  }
}
