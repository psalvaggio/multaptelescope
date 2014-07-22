// File Description
// Author: Philip Salvaggio

#ifndef HDF5_WFE_H
#define HDF5_WFE_H

#include "aperture.h"
#include "optical_designs/hdf5_wfe_parameters.pb.h"

#include <opencv/cv.h>

class Hdf5Wfe : public Aperture {
 public:
  Hdf5Wfe(const mats::SimulationConfig& params, int sim_index);

  virtual ~Hdf5Wfe();

 private:
  virtual cv::Mat GetApertureTemplate();

  virtual cv::Mat GetOpticalPathLengthDiff();

  virtual cv::Mat GetOpticalPathLengthDiffEstimate();

 private:
  Hdf5WfeParameters hdf5_wfe_params_;

  cv::Mat mask_;
  cv::Mat opd_;
  cv::Mat opd_est_;
};

#endif  // HDF5_WFE_H
