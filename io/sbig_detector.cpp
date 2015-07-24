// File Description
// Author: Philip Salvaggio

#include "sbig_detector.h"

#include <SBIGUDrv/sbigudrv.h>
#include <unistd.h>

using namespace std;
using namespace cv;

namespace mats_io {

SbigDetector::SbigDetector()
    : has_cooled_(false),
      last_error_(CE_NO_ERROR),
      error_callback_([](short, short) {}) {

  if (SendCommand(CC_OPEN_DRIVER, NULL, NULL)) {
    OpenDeviceParams device_params{DEV_USB, 0, 0};
    if (SendCommand(CC_OPEN_DEVICE, &device_params, NULL)) {
      EstablishLinkParams link_params;
      EstablishLinkResults link_results;
      SendCommand(CC_ESTABLISH_LINK, &link_params, &link_results);
    }
  }
}

SbigDetector::~SbigDetector() {
  cout << "Cleaning up detector." << endl;
  if (has_cooled_) DisableCooling();

  SBIGUnivDrvCommand(CC_CLOSE_DEVICE, NULL, NULL);
  SBIGUnivDrvCommand(CC_CLOSE_DRIVER, NULL, NULL);
}

void SbigDetector::GetSize(unsigned short& width, unsigned short& height) {
  GetCCDInfoParams query{CCD_INFO_IMAGING};
  GetCCDInfoResults0 results;
  if (SendCommand(CC_GET_CCD_INFO, &query, &results)) {
    for (int i = 0; i < results.readoutModes; i++) {
      if (results.readoutInfo[i].mode == RM_1X1) {
        width = results.readoutInfo[i].width;
        height = results.readoutInfo[i].height;
        return;
      }
    }
  }
}

void SbigDetector::Cool(double deg_c) {
  SetTemperatureRegulationParams2 temp_params{REGULATION_ON, deg_c};
  if (!SendCommand(CC_SET_TEMPERATURE_REGULATION2, &temp_params, NULL)) {
    return;
  }
  has_cooled_ = true;

  const int kMaxTime = 2 * 60 * 1e6; // [usec]
  const int kIterSleep = 5e5;
  const int kMaxIter = kMaxTime / kIterSleep;
  int iter = 0;

  bool needs_to_cool = true;
  cout << "Cooling to " << deg_c << " C" << endl;
  while (needs_to_cool && iter < kMaxIter) {
    QueryTemperatureStatusParams temp_query{TEMP_STATUS_ADVANCED2};
    QueryTemperatureStatusResults2 temp_query_results;
    if (SendCommand(CC_QUERY_TEMPERATURE_STATUS, &temp_query,
                    &temp_query_results)) {
      cout << "Temperature: " << temp_query_results.imagingCCDTemperature
           << " C (" <<  temp_query_results.fanPower << "%)      \r";
      cout.flush();
    }

    needs_to_cool = (temp_query_results.imagingCCDTemperature - deg_c) > 1;
    usleep(kIterSleep);
    iter++;
  }
  cout << endl;
}

void SbigDetector::DisableCooling() {
  const double kRoomThreshold = 15;
  SetTemperatureRegulationParams2 temp_params{REGULATION_OFF, kRoomThreshold};
  if (!SendCommand(CC_SET_TEMPERATURE_REGULATION2, &temp_params, NULL)) {
    return;
  }
  has_cooled_ = false;

  bool needs_to_warm = true;
  cout << "Warming Detector to Room Temperature" << endl;
  while (needs_to_warm) {
    QueryTemperatureStatusParams temp_query{TEMP_STATUS_ADVANCED2};
    QueryTemperatureStatusResults2 temp_query_results;
    if (SendCommand(CC_QUERY_TEMPERATURE_STATUS, &temp_query,
                    &temp_query_results)) {
      cout << "Temperature: " << temp_query_results.imagingCCDTemperature
           << " C     \r";
      cout.flush();
    }

    needs_to_warm = temp_query_results.imagingCCDTemperature -
                    kRoomThreshold < 1;
    usleep(5e5);
  }
  cout << endl;
}

void SbigDetector::Capture(unsigned short start_x,
                           unsigned short start_y,
                           unsigned short width,
                           unsigned short height,
                           unsigned long exposure_time,
                           Mat* frame) {
  if (!frame) return;

  StartExposureParams2 shot_params{
      CCD_IMAGING,
      exposure_time,
      ABG_LOW7,
      SC_OPEN_SHUTTER,
      RM_1X1,
      start_y,
      start_x,
      height,
      width};
  EndExposureParams end_params{0};
  StartReadoutParams readout_params{CCD_IMAGING,
                                    RM_1X1,
                                    start_y,
                                    start_x,
                                    height,
                                    width};
  EndReadoutParams end_readout_params{CCD_IMAGING};
  ReadoutLineParams line_params{CCD_IMAGING, RM_1X1, start_x, width};

  if (frame->cols != width || frame->rows != height ||
      frame->depth() != CV_16U) {
    frame->create(height, width, CV_16UC1);
  }
  uint16_t* frame_data = reinterpret_cast<uint16_t*>(frame->data);

  if (!SendCommand(CC_START_EXPOSURE2, &shot_params, NULL)) return;

  usleep(0.95 * exposure_time * 1e4);
  uint16_t status;
  do {
    status = GetCommandStatus(CC_START_EXPOSURE2);
  } while (status % 4 != 3);
  SendCommand(CC_END_EXPOSURE, &end_params, NULL);

  SendCommand(CC_START_READOUT, &readout_params, NULL);
  for (int i = 0; i < height; i++) {
    SendCommand(CC_READOUT_LINE, &line_params, frame_data + i * width);
  }
  SendCommand(CC_END_READOUT, &end_readout_params, NULL);
}

bool SbigDetector::has_error() const {
  return last_error_ != CE_NO_ERROR;
}

bool SbigDetector::SendCommand(short command, void* params, void* results) {
  last_error_ = SBIGUnivDrvCommand(command, params, results);
  if (last_error_ != CE_NO_ERROR) {
    GetErrorStringParams err_query{static_cast<uint16_t>(last_error_)};
    GetErrorStringResults err_string;
    SBIGUnivDrvCommand(CC_GET_ERROR_STRING, &err_query, &err_string);
    cerr << "SBIG Error (" << last_error_ << "): " << err_string.errorString
         << endl;
    error_callback_(command, last_error_);
    return false;
  }
  return true;
}

unsigned short SbigDetector::GetCommandStatus(unsigned short command) {
  QueryCommandStatusParams params{command};
  QueryCommandStatusResults result;
  SendCommand(CC_QUERY_COMMAND_STATUS, &params, &result);
  return result.status;
}


MockSbigDetector::MockSbigDetector(const Mat& image) 
    : image_(image) {}

void MockSbigDetector::Cool(double) {}
void MockSbigDetector::DisableCooling() {}
  
void MockSbigDetector::GetSize(unsigned short& width, unsigned short& height) {
  width = image_.cols;
  height = image_.rows;
}
  
void MockSbigDetector::Capture(unsigned short start_x,
                               unsigned short start_y,
                               unsigned short width,
                               unsigned short height,
                               unsigned long exposure_time,
                               Mat* frame) {
  usleep(exposure_time * 1e4);

  Mat roi;
  Range xrange(start_x, start_x + width),
        yrange(start_y, start_y + height);
  image_(yrange, xrange).copyTo(*frame);
}

void SbigAcquisitionThread(WaitQueue<Mat>* image_queue,
                           SbigDetector* detector,
                           vector<uint16_t>* roi,
                           int* exposure_time,
                           bool* keep_going) {
  while (*keep_going) {
    Mat* frame = new Mat((*roi)[3], (*roi)[2], CV_16UC1);
    detector->Capture((*roi)[0], (*roi)[1], (*roi)[2], (*roi)[3],
                      *exposure_time, frame);
    image_queue->push(frame);
  }
}

}
