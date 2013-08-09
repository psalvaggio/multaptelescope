// File Description
// Author: Philip Salvaggio

#include "system_otf.h"
#include "io/logging.h"

#include <vector>

using namespace cv;
using namespace std;

namespace mats {

SystemOtf::SystemOtf() : otf_() {}

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
    mainLog() << "Warning: Given OTF must have 1 or 2 channels." << endl;
    return;
  }

  if (otf_.rows == 0) {
    vector<Mat> new_otf_planes;
    new_otf_planes.push_back(Mat());
    new_otf_planes.push_back(Mat());

    mtf.copyTo(new_otf_planes[0]);
    ptf.copyTo(new_otf_planes[1]);

    merge(new_otf_planes, otf_);
    return;
  }

  if (otf.rows != otf_.rows || otf.cols != otf_.cols) {
    mainLog() << "Warning: All OTFs in the system must have the same array "
              << "size!" << endl;
    return;
  }

  vector<Mat> system_planes;
  split(otf_, system_planes);

  vector<Mat> new_sys_planes;
  new_sys_planes.push_back(system_planes[0] * mtf);
  new_sys_planes.push_back(system_planes[1] + ptf);
  merge(new_sys_planes, otf_);
}

}
