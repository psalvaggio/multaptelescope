// File Description
// Author: Philip Salvaggio

#include "system_otf.h"
#include "io/logging.h"

#include <vector>

using namespace cv;
using namespace std;

namespace mats {

SystemOtf::SystemOtf() : mtf_(), ptf_() {}

void SystemOtf::PushOtf(const cv::Mat& otf) {
  vector<Mat> otf_planes;
  split(otf, otf_planes);

  Mat mtf, ptf;
  if (otf_planes.size() == 1) {
    mtf = otf;
    ptf = Mat::zeros(mtf.rows, mtf.cols, CV_64FC1);
  } else if (otf_planes.size() == 2) {
    magnitude(otf_planes[0], otf_planes[1], mtf);
    phase(otf_planes[0], otf_planes[1], ptf);
  } else {
    mainLog() << "Warning: Given OTF must have 1 or 2 channels. ("
              << otf_planes.size() << " given)" << endl;
    return;
  }

  if (mtf_.rows == 0) {
    mtf_ = Mat();
    ptf_ = Mat();

    mtf.copyTo(mtf_);
    ptf.copyTo(ptf_);

    return;
  }

  if (mtf.rows != mtf_.rows || mtf.cols != mtf_.cols) {
    mainLog() << "Warning: All OTFs in the system must have the same array "
              << "size!" << endl;
    return;
  }

  mtf_ = mtf_.mul(mtf);
  ptf_ += ptf;
}

Mat SystemOtf::GetOtf() const {
  vector<Mat> otf_planes;
  otf_planes.push_back(Mat::zeros(mtf_.size(), CV_64FC1));
  otf_planes.push_back(Mat::zeros(mtf_.size(), CV_64FC1));
  
  double* mtf_data = (double*) mtf_.data;
  double* ptf_data = (double*) ptf_.data;

  double* real_data = (double*) otf_planes[0].data;
  double* imag_data = (double*) otf_planes[1].data;

  const int kSize = mtf_.rows * mtf_.cols;
  for (int i = 0; i < kSize; i++) {
    real_data[i] = mtf_data[i] * cos(ptf_data[i]);
    imag_data[i] = mtf_data[i] * sin(ptf_data[i]);
  }

  Mat otf;
  merge(otf_planes, otf);
  return otf;
}

}
