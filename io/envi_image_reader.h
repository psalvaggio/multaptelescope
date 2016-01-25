// A file reader class for ENVI images.
// Author: Philip Salvaggio

#ifndef ENVI_IMAGE_READER_H
#define ENVI_IMAGE_READER_H

#include "base/endian.h"
#include "io/envi_image_header.pb.h"

#include <cstddef>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

namespace mats_io {

std::string PrintEnviHeader(const EnviImageHeader& hdr);

// Assumes that the header file is image_filename + ".hdr".
bool EnviImread(const std::string& image_filename,
                std::vector<double>* wavelengths,
                std::vector<cv::Mat>* image);

// Reads in an ENVI image and returns it as an array of OpenCV images.
// The images will be in units of [W/m^2/sr micron^-1]. Wavelengths will be
// in units of meters.
bool EnviImread(const std::string& image_filename,
                const std::string& header_filename,
                std::vector<double>* wavelengths,
                std::vector<cv::Mat>* image);

class EnviImageReader {
 public:
  // Default constructor.
  EnviImageReader();

  // Reads in an ENVI image, assumes the header is located in the same
  // directory as the image, with the same filename, except for an additional
  // ".hdr".
  //
  // Arguments:
  //  image_filename  Full path to the image file.
  //  hdr             Output variable containing the image metadata.
  //  image           Output returned as a vector of cv::Mat objects, to avoid
  //                  dealing with interleaving issues. The wavelengths can be
  //                  found in the header object.
  bool Read(const std::string& image_filename,
            EnviImageHeader* hdr,
            std::vector<cv::Mat>* image) {
    return Read(image_filename, image_filename + ".hdr", hdr, image);
  }

  // Fully-specified Read method. Call this one if the header file does not
  // follow the naming convention above
  bool Read(const std::string& image_filename,
            const std::string& header_filename,
            EnviImageHeader* hdr,
            std::vector<cv::Mat>* image);

  // Get the wavelength units multiplier and whether an inversion is needed (if
  // wavenumber is used).
  //
  // Arguments:
  //  wave_units     The wavelength_units() string in EnviImageHeader.
  //  is_wavenumber  Output: Whether the units are wavenumbers and thus must be
  //                 inverted prior to applying the multipler.
  //
  // Returns:
  //  The conversion multiplier between the wavelength units and meters.
  static double GetWavelengthMultiplier(const std::string& wave_units,
                                       bool* is_wavenumber);

 private:
  // Parse the header file and populate the EnviImageHeader object.
  //
  // Arguments:
  //  header_filename  The filename of the ENVI header file
  //  hdr              Output: EnviImageHeader to hold all of the metadata
  bool ReadHeader(const std::string& header_filename,
                  EnviImageHeader* hdr);

  // Given a line from the header file, populate the appropriate header field.
  // All multiline definitions are guaranteed to be collapsed into one line
  // before this function is called, so all lines follow the format:
  //   tag_name = value
  //
  // Arguments:
  //  line  The line from the header file (may be a collapse of multiple lines)
  //  hdr   Output: The EnviImageHeader to populate
  bool ParseHeaderLine(const std::string& line, EnviImageHeader* hdr);

  // Convert ENVI data types to OpenCV data types.
  //
  // Arguments:
  //  envi_dt  The ENVI data type parsed from the header file.
  //  size     Output: The number of bytes in the data type
  //
  // Returns:
  //  The OpenCV data type, or -1 if the ENVI data type is not supported.
  int GetOpenCVDType(int envi_dt, int& size);

  // Calculate the 1-D index for a pixel given the row, column, band, and
  // header information
  template <class T>
  void RearrangeImageData(T* data,
                          EnviImageHeader* hdr,
                          std::vector<cv::Mat>* image);
};

template <class T>
void EnviImageReader::RearrangeImageData(T* data,
                                        EnviImageHeader* hdr,
                                        std::vector<cv::Mat>* image) {
  const size_t kBands = hdr->bands();
  const size_t kRows = hdr->lines();
  const size_t kCols = hdr->samples();

  bool needs_endian_swap = (hdr->byte_order() == 0 && !mats::isLittleEndian())
                        || (hdr->byte_order() == 1 && mats::isLittleEndian());


  size_t band_stride = 0, row_stride = 0, col_stride = 0;
  switch (hdr->interleave_type()) {
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
    T* band_data = (T*)image->at(b).data;

    for (size_t r = 0; r < kRows; r++) {
      for (size_t c = 0; c < kCols; c++) {
        T val = data[b * band_stride + r * row_stride + c * col_stride];
        if (needs_endian_swap)  val = mats::swap_endian(val);
        band_data[r * kCols + c] = val;
      }
    }
  }
}

}

#endif  // ENVI_IMAGE_READER_H
