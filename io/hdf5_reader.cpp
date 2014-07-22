// File Description
// Author: Philip Salvaggio

#include "hdf5_reader.h"

#include "io/logging.h"

#include <hdf5.h>
#include <hdf5_hl.h>

using namespace std;

namespace mats_io {

bool HDF5Reader::Read(const string& filename,
                      const string& dataset,
                      cv::Mat* data) {
  hid_t loc_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (loc_id < 0) {
    mainLog() << "Error: Could not open HDF5 file " << filename << endl;
    return false;
  }

  htri_t valid = H5LTpath_valid(loc_id, dataset.c_str(), true);
  if (!valid || valid < 0) {
    mainLog() << "Error: Invalid dataset " << dataset << " in HDF5 file."
              << endl;
    return false;
  }

  H5T_class_t class_id;
  hsize_t wfe_dims[2];
  size_t dtype_size;
  if (H5LTget_dataset_info(loc_id, dataset.c_str(),
        wfe_dims, &class_id, &dtype_size) < 0) {
    mainLog() << "Error: Could not query dataset size." << endl;
    return false;
  }

  const size_t kSize = wfe_dims[0] * wfe_dims[1];
  float* buffer = new float[kSize];
  if (H5LTread_dataset_float(loc_id, dataset.c_str(), buffer) < 0) {
    mainLog() << "Error: Could not read the dataset." << endl;
    return false;
  }

  data->create(wfe_dims[0], wfe_dims[1], CV_32FC1);
  float* data_ptr = (float*)data->data;
  memcpy(data_ptr, buffer, sizeof(float) * kSize);

  cv::transpose(*data, *data);

  H5Fclose(loc_id);

  return true;
}

}
