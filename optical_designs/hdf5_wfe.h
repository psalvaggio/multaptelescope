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
  void GetApertureTemplate(cv::Mat_<double>* output) const override;

  void GetOpticalPathLengthDiff(cv::Mat_<double>* output) const override;

  void GetOpticalPathLengthDiffEstimate(
      cv::Mat_<double>* output) const override;

 private:
  Hdf5WfeParameters hdf5_wfe_params_;
};
REGISTER_APERTURE(Hdf5Wfe, HDF5_WFE)

#endif  // HDF5_WFE_H
