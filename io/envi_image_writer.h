// A file writer class for ENVI images.
// Author: Philip Salvaggio

#ifndef ENVI_IMAGE_WRITER_H
#define ENVI_IMAGE_WRITER_H

#include "base/endian.h"
#include "base/macros.h"
#include "io/envi_image_header.pb.h"

#include <cstddef>
#include <string>
#include <vector>
#include <opencv/cv.h>

namespace mats_io {

class EnviImageWriter {
 public:
  // Default constructor.
  EnviImageWriter();

  // Writes an ENVI image, assumes the header is located in the same
  // directory as the image, with the same filename, except for an additional
  // ".hdr".
  //
  // Arguments:
  //  image_filename  Full path to the image file.
  //  hdr             Header with image metadata. Will be augmented by this
  //                  routine.
  //  image           A vector of cv::Mat objects representing the image. The
  //                  band information can be found in the header.
  bool Write(const std::string& image_filename,
             EnviImageHeader* hdr,
             const std::vector<cv::Mat>& image) {
    return Write(image_filename, image_filename + ".hdr", hdr, image);
  }

  // Fully-specified Write method. Call this one if the header file does not
  // follow the naming convention above
  bool Write(const std::string& image_filename,
             const std::string& header_filename,
             EnviImageHeader* hdr,
             const std::vector<cv::Mat>& image);

 private:
  // Write out the ENVI image header.
  //
  // Arguments:
  //  header_filename  The filename of the ENVI header file
  //  hdr              Output: EnviImageHeader to hold all of the metadata
  bool WriteHeader(const std::string& header_filename,
                   const EnviImageHeader& hdr);

  // Convert OpenCV data types to ENVI data types.
  //
  // Arguments:
  //  opencv_dt  The OpenCV data type to be converted.
  //  size       Output: The number of bytes in the data type
  //
  // Returns:
  //  The ENVI data type, or -1 if the OpenCV data type is not supported.
  int GetEnviDType(int opencv_dtype, int& size);

  // Calculate the 1-D index for a pixel given the row, column, band, and
  // header information
  template <class T>
  void RearrangeImageData(const std::vector<cv::Mat>& image,
                          const EnviImageHeader& hdr,
                          T* data);

 NO_COPY_OR_ASSIGN(EnviImageWriter)
};

template <class T>
void EnviImageWriter::RearrangeImageData(const std::vector<cv::Mat>& image,
                                         const EnviImageHeader& hdr,
                                         T* data) {
  const size_t kBands = hdr.bands();
  const size_t kRows = hdr.lines();
  const size_t kCols = hdr.samples();

  bool needs_endian_swap = (hdr.byte_order() == 0 && !mats::isLittleEndian())
                        || (hdr.byte_order() == 1 && mats::isLittleEndian());


  size_t band_stride = 0, row_stride = 0, col_stride = 0;
  switch (hdr.interleave_type()) {
    case EnviImageHeader::BSQ:
      band_stride = kRows * kCols;
      row_stride = kCols;
      col_stride = 1;
      break;
    case EnviImageHeader::BIL:
      band_stride = kCols;
      row_stride = kCols * kBands;
      col_stride = 1;
      break;
    case EnviImageHeader::BIP:
      band_stride = 1;
      row_stride = kBands * kCols;
      col_stride = kBands;
      break;
  }

  for (size_t b = 0; b < kBands; b++) {
    const T* band_data = (const T*)image[b].data;

    for (size_t r = 0; r < kRows; r++) {
      for (size_t c = 0; c < kCols; c++) {
        T val = band_data[r * kCols + c];
        if (needs_endian_swap)  val = mats::swap_endian(val);
        data[b * band_stride + r * row_stride + c * col_stride] = val;
      }
    }
  }
}

}

#endif  // ENVI_IMAGE_WRITER_H
