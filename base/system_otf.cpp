// File Description
// Author: Philip Salvaggio

#include "system_otf.h"
#include "io/logging.h"

#include <vector>

using namespace cv;
using namespace std;

namespace mats {

SystemOtf::SystemOtf() : otf_() {}

void SystemOtf::PushOtf(const cv::Mat_<double>& mtf) {
  if (otf_.rows == 0) {
    vector<Mat> otf_planes;
    otf_planes.push_back(mtf);
    otf_planes.push_back(Mat::zeros(mtf.size(), CV_64FC1));
    merge(otf_planes, otf_);
  } else {
    vector<Mat> otf_planes;
    split(otf_, otf_planes);
    for (auto& plane : otf_planes) multiply(plane, mtf, plane);
    merge(otf_planes, otf_);
  }
}

void SystemOtf::PushOtf(const cv::Mat_<complex<double>>& otf) {
  if (otf_.rows == 0) {
    otf.copyTo(otf_);
  } else {
    mulSpectrums(otf_, otf, otf, 0);
  }
}

}
