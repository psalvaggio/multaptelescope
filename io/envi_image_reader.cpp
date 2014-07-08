// A file reader class for ENVI images.
// Author: Philip Salvaggio

#include "envi_image_reader.h"

#include "io/envi_image_header.pb.h"
#include "io/logging.h"
#include "base/endian.h"
#include "base/str_utils.h"

#include <fstream>
#include <sstream>
#include <cstdlib>

using namespace std;
using mats::trim;

namespace mats_io {

std::string PrintEnviHeader(const EnviImageHeader& hdr) {
  stringstream ss;

  ss << "ENVI Image Header" << endl
     << "  Description: " << hdr.description() << endl
     << "  Samples: " << hdr.samples() << endl
     << "  Lines: " << hdr.lines() << endl
     << "  Bands: " << hdr.bands() << endl
     << "  Data Type: " << hdr.data_type() << endl
     << "  Interleave Type: " << hdr.interleave_type() << endl
     << "  Byte Order: " << hdr.byte_order() << endl
     << "  Sensor Type: " << hdr.sensor_type() << endl
     << "  Wavelength Units: " << hdr.wavelength_units() << endl;

  for (int i = 0; i < hdr.band_size(); i++) {
    if (hdr.band(i).bad_band_multiplier() == 0) continue;

    ss << "  Band " << (i+1) << ":" << endl
       << "    Name: " << hdr.band(i).name() << endl
       << "    Center Wavelength: " << hdr.band(i).center_wavelength() << endl
       << "    FWHM: " << hdr.band(i).fwhm() << endl;
  }

  return ss.str();
}

EnviImageReader::EnviImageReader() {}

bool EnviImageReader::Read(const string& image_filename,
                           const string& header_filename,
                           EnviImageHeader* hdr,
                           vector<cv::Mat>* image) {
  // Validate the input.
  if (!hdr || !image) {
    mainLog() << "Invalid header or image pointer" << endl;
    return false;
  }
  image->clear();

  // Read in the ENVI header file.
  if (!ReadHeader(header_filename, hdr)) {
    mainLog() << "Error: Invalid ENVI header file \"" << header_filename
              << "\"" << endl;
    return false;
  }

  if (!hdr->has_byte_order()) {
    mainLog() << "Error: No byte order given in ENVI header file \""
              << header_filename << "\"" << endl;
    return false;
  }

  if (!hdr->has_data_type()) {
    mainLog() << "Error: ENVI header \"" << header_filename
              << "\" missing data type." << endl;
    return false;
  }
  
  int depth = 0;
  int cv_type = GetOpenCVDType(hdr->data_type(), depth);
  if (cv_type == -1) return false;

  // Size of the data.
  const int kSize = hdr->lines() * hdr->samples() * hdr->bands() * depth;

  if (!hdr->has_bands() || !hdr->has_samples() || !hdr->has_lines() ||
      hdr->bands() <= 0 || hdr->samples() <= 0 || hdr->lines() <= 0) {
    mainLog() << "Error: Missing or invalid image dimensions in ENVI header "
              << "file \"" << header_filename << "\"" << endl;
    return false;
  }

  if (hdr->band_size() != hdr->bands()) {
    mainLog() << "The number of bands in the ENVI header does not match up "
              << "with the number of bands described in the \"band names\"/"
              << "wavelengths/... sections." << endl;
    return false;
  }

  // Allocate the memory for the bands;
  for (int i = 0; i < hdr->bands(); i++) {
    image->push_back(cv::Mat(hdr->lines(), hdr->samples(), cv_type));
  }

  // Open the image file.
  FILE* img_file = fopen(image_filename.c_str(), "rb");
  if (!img_file) {
    mainLog() << "Can not open image file \"" << image_filename
              << "\"" << endl;
    return false;
  }

  if (fseek(img_file, hdr->header_offset(), SEEK_SET) != 0) {
    mainLog() << "The header offset in the ENVI header \"" << header_filename
              << "\" was invalid." << endl;
    return false;
  }

  unsigned char* raw_data = new unsigned char[kSize];
  if (fread(raw_data, 1, kSize, img_file) != kSize) {
    mainLog() << "Not enough data in the image file." << endl;
    delete[] raw_data;
    return false;
  }
  fclose(img_file);

  switch (cv_type) {
    case CV_8UC1: RearrangeImageData(raw_data, hdr, image); break;
    case CV_16SC1: RearrangeImageData((int16_t*)raw_data, hdr, image); break;
    case CV_32SC1: RearrangeImageData((int32_t*)raw_data, hdr, image); break;
    case CV_32FC1: RearrangeImageData((float*)raw_data, hdr, image); break;
    case CV_64FC1: RearrangeImageData((double*)raw_data, hdr, image); break;
    case CV_16UC1: RearrangeImageData((uint16_t*)raw_data, hdr, image); break;
  }

  delete[] raw_data;

  return true;
}

bool EnviImageReader::ReadHeader(const string& header_filename,
                                 EnviImageHeader* hdr) {
  ifstream ifs(header_filename.c_str());

  if (!ifs.is_open()) {
    mainLog() << "Could not open file: " << header_filename << endl;
    return false;
  }
  
  string line, multiline;

  getline(ifs, line);
  if (line != "ENVI") {
    mainLog() << "First line of ENVI header file should be \"ENVI\"" << endl;
    return false;
  }

  getline(ifs, line);
  multiline = line;

  int line_num = 3;
  while (!ifs.eof()) {
    getline(ifs, line);

    if (line.find_first_of("=") != string::npos) {
      if (!ParseHeaderLine(multiline, hdr)) {
        mainLog() << "Invalid ENVI header line encountered at line "
                  << line_num << endl;
        return false;
      }
           
      multiline = "";
    }

    multiline.append(trim(line));
    line_num++;
  }

  if (!ParseHeaderLine(multiline, hdr)) {
    mainLog() << "Invalid ENVI header line encountered at line "
              << line_num << endl;
    return false;
  }

  return true;
}

bool EnviImageReader::ParseHeaderLine(const string& line,
                                      EnviImageHeader* hdr) {
  int equal_index = line.find_first_of("=");
  if (equal_index == string::npos) {
    mainLog() << "No \"=\" found in the line." << endl;
    return false;
  }

  string tag_name = line.substr(0, equal_index);
  trim(tag_name);
  std::transform(tag_name.begin(), tag_name.end(), tag_name.begin(), ::tolower);
  string value = line.substr(equal_index + 1);
  trim(value);

  if (tag_name == "description") {
    hdr->set_description(value);
  } else if (tag_name == "samples") {
    hdr->set_samples(atoi(value.c_str()));
  } else if (tag_name == "lines") {
    hdr->set_lines(atoi(value.c_str()));
  } else if (tag_name == "bands") {
    hdr->set_bands(atoi(value.c_str()));
  } else if (tag_name == "header offset") {
    hdr->set_header_offset(atoi(value.c_str()));
  } else if (tag_name == "file type") {
    hdr->set_file_type(value);
  } else if (tag_name == "data type") {
    hdr->set_data_type(atoi(value.c_str()));
  } else if (tag_name == "interleave") {
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    if (value == "bip") {
      hdr->set_interleave_type(EnviImageHeader::BIP);
    } else if (value == "bil") {
      hdr->set_interleave_type(EnviImageHeader::BIL);
    } else if (value == "bsq") {
      hdr->set_interleave_type(EnviImageHeader::BSQ);
    } else {
      mainLog() << "Invalid interleave type given in ENVI header." << endl;
      return false;
    }
  } else if (tag_name == "sensor type") {
    hdr->set_sensor_type(value);
  } else if (tag_name == "byte order") {
    hdr->set_byte_order(atoi(value.c_str()));
  } else if (tag_name == "x start") {
    hdr->set_start_x(atoi(value.c_str()));
  } else if (tag_name == "y start") {
    hdr->set_start_y(atoi(value.c_str()));
  } else if (tag_name == "map info") {
    hdr->set_map_info(value);
  } else if (tag_name == "projection info") {
    hdr->set_projection_info(value);
  } else if (tag_name == "default bands") {
    hdr->set_default_bands(value);
  } else if (tag_name == "z plot range") {
    hdr->set_z_plot_range(value);
  } else if (tag_name == "z plot average") {
    hdr->set_z_plot_average(value);
  } else if (tag_name == "z plot titles") {
    hdr->set_z_plot_titles(value);
  } else if (tag_name == "pixel size") {
    hdr->set_pixel_size(value.c_str());
  } else if (tag_name == "default stretch") {
    hdr->set_default_stretch(value);
  } else if (tag_name == "wavelength units") {
    hdr->set_wavelength_units(value);
  } else if (tag_name == "band names") {
    vector<string> band_names;
    mats::explode(value, ',', &band_names);

    if (hdr->band_size() > 0 && hdr->band_size() != band_names.size()) {
      mainLog() << "Wavelengths, FWHM, band names and bad bands list all "
                << "need to be the same length." << endl;
      return false;
    }

    if (hdr->band_size() == 0) {
      for (size_t i = 0; i < band_names.size(); i++) {
        EnviBand* band = hdr->add_band();
        band->set_name(band_names[i]);
      }
    } else {
      for (size_t i = 0; i < band_names.size(); i++) {
        hdr->mutable_band(i)->set_name(band_names[i]);
      }
    }
  } else if (tag_name == "wavelength") {
    vector<string> wavelengths;
    mats::explode(value, ',', &wavelengths);
    if (wavelengths.size() > 0 && wavelengths[0].substr(0, 1) == "{") {
      wavelengths[0] = wavelengths[0].substr(1);
    }

    if (hdr->band_size() > 0 && hdr->band_size() != wavelengths.size()) {
      mainLog() << "Wavelengths, FWHM, band names and bad bands list all "
                << "need to be the same length." << endl;
      return false;
    }

    if (hdr->band_size() == 0) {
      for (size_t i = 0; i < wavelengths.size(); i++) {
        EnviBand* band = hdr->add_band();
        band->set_center_wavelength(atof(trim(wavelengths[i]).c_str()));
      }
    } else {
      for (size_t i = 0; i < wavelengths.size(); i++) {
        hdr->mutable_band(i)->set_center_wavelength(
            atof(trim(wavelengths[i]).c_str()));
      }
    }
  } else if (tag_name == "fwhm") {
    vector<string> fwhms;
    mats::explode(value, ',', &fwhms);
    if (fwhms.size() > 0 && fwhms[0].substr(0, 1) == "{") {
      fwhms[0] = fwhms[0].substr(1);
    }

    if (hdr->band_size() > 0 && hdr->band_size() != fwhms.size()) {
      mainLog() << "Wavelengths, FWHM, band names and bad bands list all "
                << "need to be the same length." << endl;
      return false;
    }

    if (hdr->band_size() == 0) {
      for (size_t i = 0; i < fwhms.size(); i++) {
        EnviBand* band = hdr->add_band();
        band->set_fwhm(atof(trim(fwhms[i]).c_str()));
      }
    } else {
      for (size_t i = 0; i < fwhms.size(); i++) {
        hdr->mutable_band(i)->set_fwhm(atof(trim(fwhms[i]).c_str()));
      }
    }

  } else if (tag_name == "bbl") {
    vector<string> bbls;
    mats::explode(value, ',', &bbls);
    if (bbls.size() > 0 && bbls[0].substr(0, 1) == "{") {
      bbls[0] = bbls[0].substr(1);
    }

    if (hdr->band_size() > 0 && hdr->band_size() != bbls.size()) {
      mainLog() << "Wavelengths, FWHM, band names and bad bands list all "
                << "need to be the same length." << endl;
      return false;
    }

    if (hdr->band_size() == 0) {
      for (size_t i = 0; i < bbls.size(); i++) {
        EnviBand* band = hdr->add_band();
        band->set_bad_band_multiplier(atof(trim(bbls[i]).c_str()));
      }
    } else {
      for (size_t i = 0; i < bbls.size(); i++) {
        hdr->mutable_band(i)->set_bad_band_multiplier(
            atof(trim(bbls[i]).c_str()));
      }
    }
  } else {
    mainLog() << "Unrecognized tag name \"" << tag_name << "\"" << endl;
    return false;
  }

  return true;
}

int EnviImageReader::GetOpenCVDType(int envi_dt, int& size) {
  switch (envi_dt) {
    case EnviImageHeader::BYTE8: size = 1; return CV_8UC1;
    case EnviImageHeader::INT16: size = 2; return CV_16SC1;
    case EnviImageHeader::INT32: size = 4; return CV_32SC1;
    case EnviImageHeader::FLOAT32: size = 4; return CV_32FC1;
    case EnviImageHeader::FLOAT64: size = 8; return CV_64FC1;
    case EnviImageHeader::COMPLEX2x32:
    case EnviImageHeader::COMPLEX2x64:
      mainLog() << "Error: Complex inputs not yet supported." << endl;
      break;
    case EnviImageHeader::UINT16: size = 2; return CV_16UC1;
    case EnviImageHeader::UINT32:
    case EnviImageHeader::UINT64:
    case EnviImageHeader::UINT64_LONG:
      mainLog() << "Error: Unsupported data type in ENVI header." << endl;
      break;
    default:
      mainLog() << "Error: Invalid data type in ENVI header." << endl;
      break;
  }

  return -1;
}

}
