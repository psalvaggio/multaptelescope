// A class in which to accumulate the optical transfer function (OTF) of a
// system. The OTF is the Fourier transform of the system's point spread
// function (PSF). It's magnitude is the Modulation Transfer Function (MTF),
// which describes how contrast is lost at each spatial frequency in the
// system. The phase of the OTF is the Phase Transfer Function and describes
// any phase shifts that occur at each spatial frequency, which can result in
// contrast reversals.
//
// Author: Philip Salvaggio

#ifndef SYSTEM_OTF_H
#define SYSTEM_OTF_H

#include <opencv/cv.h>

namespace mats {

class SystemOtf {
 public:
  SystemOtf();

  void PushOtf(const cv::Mat& otf);

  cv::Mat GetOtf() const;

 private:
  cv::Mat mtf_, ptf_;
};

}

#endif  // SYSTEM_OTF_H
