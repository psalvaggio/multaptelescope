// File Description
// Author: Philip Salvaggio

#include "hdf5_wfe.h"

#include "base/zernike_aberrations.h"
#include "base/assertions.h"
#include "io/hdf5_reader.h"
#include "io/logging.h"

#include <opencv2/opencv.hpp>

using namespace cv;

namespace mats {

Hdf5Wfe::Hdf5Wfe(const Simulation& params)
    : Aperture(params) {
  hdf5_wfe_params_ = aperture_params().GetExtension(hdf5_wfe_params);
}

Hdf5Wfe::~Hdf5Wfe() {}

void Hdf5Wfe::GetApertureTemplate(Mat_<double>* output) const {
  Mat_<double> mask = *output;

  const int kSize = mask.rows;

  Mat_<double> opd = GetWavefrontError(kSize, 0, 0);

  CHECK(mask.rows == opd.rows && mask.cols == opd.cols);

  for (int i = 0; i < kSize; i++) {
    for (int j = 0; j < kSize; j++) {
      mask(i, j) = (opd(i, j) != 0) ? 1 : 0;
    }
  }
}

void Hdf5Wfe::GetOpticalPathLengthDiff(double /*image_height*/,
                                       double /*angle*/,
                                       Mat_<double>* output) const {
  Mat opd;
  CHECK(mats_io::HDF5Reader::Read(hdf5_wfe_params_.wfe_filename(),
                                  hdf5_wfe_params_.dataset(),
                                  &opd));

  cv::threshold(opd, opd, hdf5_wfe_params_.background_value(), 0,
                cv::THRESH_TOZERO_INV);
  opd.convertTo(opd, CV_64F);

  Mat col_sums;
  cv::reduce(opd, col_sums, 0, CV_REDUCE_SUM);
  double* sum_data = reinterpret_cast<double*>(col_sums.data);
  int first_col = 0, last_col = opd.cols - 1;
  for (; sum_data[first_col] == 0 && first_col < opd.cols; first_col++) {}
  for (; sum_data[last_col] == 0 && last_col >= 0; last_col--) {}

  Mat row_sums;
  cv::reduce(opd, row_sums, 1, CV_REDUCE_SUM);
  sum_data = reinterpret_cast<double*>(row_sums.data);
  int first_row = 0, last_row = opd.rows - 1;
  for (; sum_data[first_row] == 0 && first_row < opd.rows; first_row++) {}
  for (; sum_data[last_row] == 0 && last_row >= 0; last_row--) {}

  if (last_col < first_col || last_row < first_row) {
    mainLog() << "Error: Given wavefront error file was blank." << std::endl;
    exit(1);
  }

  cv::Range row_range(first_row, last_row + 1),
            col_range(first_col, last_col + 1);
  cv::resize(opd(row_range, col_range), *output, output->size());
}

}
