// A hardware interface class for an SBIG detector.
// Author: Philip Salvaggio

#ifndef SBIG_DETECTOR_H
#define SBIG_DETECTOR_H

#include "base/wait_queue.h"

#include <opencv2/core/core.hpp>

namespace mats_io {

class SbigDetector {
 public:
  // Opens the driver and device, establishes a link to the camera. The camera
  // is assumed to be connected via USB. If you have another connection type or
  // don't want to do a USB auto-detect scan, modify the body of the
  // constructor.
  SbigDetector();

  SbigDetector(const SbigDetector&) = delete;
  SbigDetector& operator=(const SbigDetector&) = delete;

  // Closes the driver, blacks and warms the detector if cooling was enabled.
  ~SbigDetector();

  // Cool and warm the detector.
  virtual void Cool(double deg_c);
  virtual void DisableCooling();

  // Get the full size of the imaging CCD.
  virtual void GetSize(unsigned short& width, unsigned short& height);

  // Capture a frame into an OpenCV image. Exposure time is in hundreths of a
  // second.
  virtual void Capture(unsigned short start_x,
                       unsigned short start_y,
                       unsigned short width,
                       unsigned short height,
                       unsigned long exposure_time,
                       cv::Mat* frame);

  // Get the last error from the detector.
  int last_error() const { return last_error_; }
  bool has_error() const;

  // This function will be called on any error, the first parameter is the
  // command ID and the second is the error ID. This class will dump the error
  // string to cerr, so this is only necessary if you want to exit on an error.
  using ErrorCallback = std::function<void(short, short)>;
  void set_error_callback(const ErrorCallback& callback) {
    error_callback_ = callback;
  }

 private:
  bool SendCommand(short command, void* params, void* results);
  unsigned short GetCommandStatus(unsigned short command);
  
 private:
  bool has_cooled_;
  int last_error_;
  ErrorCallback error_callback_;
};

void SbigAcquisitionThread(WaitQueue<cv::Mat>* image_queue,
                           SbigDetector* detector,
                           std::vector<uint16_t>* roi,
                           int* exposure_time,
                           bool* keep_going);

class MockSbigDetector : public SbigDetector {
 public:
  MockSbigDetector(const cv::Mat& image);

  // Cool and warm the detector.
  void Cool(double deg_c) override;
  void DisableCooling() override;
  
  // Get the full size of the imaging CCD.
  void GetSize(unsigned short& width, unsigned short& height) override;
  
  // Capture a frame into an OpenCV image. Exposure time is in hundreths of a
  // second.
  void Capture(unsigned short start_x,
               unsigned short start_y,
               unsigned short width,
               unsigned short height,
               unsigned long exposure_time,
               cv::Mat* frame) override;
 
 private:
  cv::Mat image_;
};

}

#endif  // SBIG_DETECTOR_H
