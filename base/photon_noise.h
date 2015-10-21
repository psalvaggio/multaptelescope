// A class for adding photon noise to data.
// Author: Philip Salvaggio

#ifndef PHOTON_NOISE_H
#define PHOTON_NOISE_H

#include <opencv2/core/core.hpp>

namespace mats {

class PhotonNoise {
 public:
  PhotonNoise();
  ~PhotonNoise();

  void AddPhotonNoise(cv::Mat* signal);
  void AddPhotonNoise(const cv::Mat& input, cv::Mat* output);
};

}

#endif  // PHOTON_NOISE_H
