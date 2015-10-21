// File Description
// Author: Philip Salvaggio

#ifndef HDF5_WFE_H
#define HDF5_WFE_H

#include "aperture.h"
#include "optical_designs/hdf5_wfe_parameters.pb.h"

#include <opencv2/core/core.hpp>

class Hdf5Wfe : public Aperture {
 public:
  explicit Hdf5Wfe(const mats::Simulation& params);

  virtual ~Hdf5Wfe();

 private:
  void GetApertureTemplate(cv::Mat_<double>* output) const override;

  void GetOpticalPathLengthDiff(double image_height,
                                double angle,
                                cv::Mat_<double>* output) const override;

 private:
  Hdf5WfeParameters hdf5_wfe_params_;
};
REGISTER_APERTURE(Hdf5Wfe, HDF5_WFE)

#endif  // HDF5_WFE_H
