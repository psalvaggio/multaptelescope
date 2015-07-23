// File Description
// Author: Philip Salvaggio

#include "zernike_aberrations.h"
#include "zernike_cuda.h"

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

  const size_t kSize = output_size;
  const double kCenter = 0.5 * (output_size - 1);

  if (output->rows != output_size || output->cols != output_size) {
    output->create(output_size, output_size, CV_64F);
  }
  *output = Scalar(0);

  for (size_t i = 0; i < kSize; i++) {
    double y = i - kCenter;
    for (size_t j = 0; j < kSize; j++) {
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
          case 9:
            wfe += weights[9] * rho * rho * rho * cos(3 * theta); break;
          case 10:
            wfe += weights[10] * rho * rho * rho * sin(3 * theta); break;
          case 11:
            wfe += weights[11] * rho * rho * (4 * rho * rho - 3) *
                   cos(2 * theta);
            break;
          case 12:
            wfe += weights[12] * rho * rho * (4 * rho * rho - 3) *
                   sin(2 * theta);
            break;
          case 13:
            wfe += weights[13] * rho * (3 - 12 * rho*rho + 10 * pow(rho, 4)) *
                   cos_theta;
            break;
          case 14:
            wfe += weights[14] * rho * (3 - 12 * rho*rho + 10 * pow(rho, 4)) *
                   sin_theta;
            break;
          case 15:
            wfe += weights[15] * (-1 + 12 * rho*rho - 30 * pow(rho, 4) + 20 *
                   pow(rho, 6));
            break;
          case 16:
            wfe += weights[16] * pow(rho, 4) * cos(4 * theta); break;
          case 17:
            wfe += weights[17] * pow(rho, 4) * sin(4 * theta); break;
          case 18:
            wfe += weights[18] * pow(rho, 3) * (5 * rho*rho - 4) *
                   cos(3 * theta);
            break;
          case 19:
            wfe += weights[19] * pow(rho, 3) * (5 * rho*rho - 4) *
                   sin(3 * theta);
            break;
          case 20:
            wfe += weights[20] * rho * rho * (6 - 20 * rho*rho + 15 *
                   pow(rho, 4)) * cos(2 * theta);
            break;
          case 21:
            wfe += weights[21] * rho * rho * (6 - 20 * rho*rho + 15 *
                   pow(rho, 4)) * sin(2 * theta);
            break;
          case 22:
            wfe += weights[22] * rho * (-4 + 30 * rho*rho - 60 * pow(rho, 4) +
                   35 * pow(rho, 6)) * cos(theta);
            break;
          case 23:
            wfe += weights[23] * rho * (-4 + 30 * rho*rho - 60 * pow(rho, 4) +
                   35 * pow(rho, 6)) * sin(theta);
            break;
          case 24:
            wfe += weights[24] * (1 - 20 * rho*rho + 90 * pow(rho, 4) - 140 *
                   pow(rho, 6) + 70 * pow(rho, 8));
            break;
        }
      }
      output->at<double>(i, j) = wfe;
    }
  }
}
