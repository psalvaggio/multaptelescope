// File Description
// Author: Philip Salvaggio

#include "mats.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <gflags/gflags.h>

DEFINE_string(config_file, "", "SimulationConfig filename.");
DEFINE_string(target_file, "", "Target image filename.");
DEFINE_double(smoothness, 1e-2, "Inverse filtering smoothness.");
DEFINE_double(orientation, 0, "Aperture orientation.");
DEFINE_double(full_well_frac, 0.8, "Fraction of the full-well capacity for the"
                                   " bright regions.");
DEFINE_double(peak_coma, 0.5, "The peak value of coma.");
DEFINE_int32(radial_zones, 5, "The number of isoplanatic regions in the radial "
                              "dimension.");
DEFINE_int32(angular_zones, 12, "The number of isoplanatic regions in the "
                                "angular dimension.");

using namespace std;
using namespace cv;
using namespace mats;

static const double h = 6.62606957e-34;  // [J*s]
static const double c = 299792458;  // [m/s]

void CorrectExposure(const Mat& image,
                     const vector<double>& wavelengths,
                     const vector<double>& illumination,
                     const Telescope& telescope,
                     vector<Mat>* output) {
  const Detector& det = *(telescope.detector());

  // Compute the central wavelength
  double central_wavelength = 0;
  for (size_t i = 0; i < wavelengths.size(); i++) {
    central_wavelength += wavelengths[i] * illumination[i];
  }

  // Do some radiometry to determine the correct exposure for the telescope.
  // Compute the desired irradiance [W/m^2] incident upon the detector.
  double target_electrons = FLAGS_full_well_frac * det.full_well_capacity();
  double eff_qe = det.GetEffectiveQE(wavelengths, illumination, 0);
  double target_photons = target_electrons / eff_qe;
  double int_time = telescope.simulation().integration_time();
  double target_flux = target_photons * h * c / central_wavelength;
  double target_irradiance = target_flux / (det.detector_area() * int_time);

  // Compute the effective G/# to convert the detector irradiance to
  // entrance-pupil reaching radiance.
  double eff_g_num = 0;
  for (size_t i = 0; i < illumination.size(); i++) {
    eff_g_num += illumination[i] * telescope.GNumber(wavelengths[i]);
  }

  // Construct the hyperspectral input at the proper light level
  output->clear();
  for (size_t i = 0; i < wavelengths.size(); i++) {
    output->push_back(image * target_irradiance * eff_g_num * illumination[i]);
  }
}

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  // Read in the input model parameters
  mats::SimulationConfig sim_config;
  mats::DetectorParameters detector_params;
  if (!MatsInit(ResolvePath(FLAGS_config_file),
                &sim_config,
                &detector_params,
                nullptr, nullptr)) {
    return 1;
  }

  // Read in the spectral weighting function for the binary target
  vector<vector<double>> data;
  if (!mats_io::TextFileReader::Parse(sim_config.spectral_weighting_filename(),
                                      &data)) {
    cerr << "Could not read spectral weighting file." << endl;
    return 1;
  }
  const vector<double>& wavelengths = data[0];
  vector<double>& illumination = data[1];

  // Normalize the illumination function
  double total_weight = accumulate(begin(illumination), end(illumination), 0.);
  for (auto& tmp : illumination) tmp /= total_weight;

  // Read in the input image and peak normalize
  Mat image = imread(ResolvePath(FLAGS_target_file), 0);
  if (!image.data) {
    cerr << "Could not read image file." << endl;
    return 1;
  }
  image.convertTo(image, CV_64F);
  double im_max;
  minMaxLoc(image, NULL, &im_max);
  image /= im_max;

  // Set the size of the detector
  detector_params.set_array_rows(image.rows);
  detector_params.set_array_cols(image.cols);

  // Perform each simulation
  for (int i = 0; i < sim_config.simulation_size(); i++) {
    // Clear out existing coma and replace it
    ApertureParameters* ap_params =
        sim_config.mutable_simulation(i)->mutable_aperture_params();
    ZernikeCoefficient* coma_x = nullptr;
    ZernikeCoefficient* coma_y = nullptr;
    for (int j = 0; j < ap_params->aberration_size(); j++) {
      if (ap_params->aberration(j).type() == ZernikeCoefficient::COMA_X) {
        coma_x = ap_params->mutable_aberration(j);
      } else if (ap_params->aberration(j).type() ==
                 ZernikeCoefficient::COMA_Y) {
        coma_y = ap_params->mutable_aberration(j);
      }
    }
    if (coma_x == nullptr) {
      coma_x = ap_params->add_aberration();
      coma_x->set_type(ZernikeCoefficient::COMA_X);
    }
    if (coma_y == nullptr) {
      coma_y = ap_params->add_aberration();
      coma_y->set_type(ZernikeCoefficient::COMA_Y);
    }
    coma_x->set_value(0);
    coma_y->set_value(0);

    mainLog() << "Simulation " << (i+1) << " of "
              << sim_config.simulation_size() << endl
              << mats_io::PrintSimulation(sim_config.simulation(i))
              << mats_io::PrintAperture(
                  sim_config.simulation(i).aperture_params());

    // Apply the command line flag for orientation
    sim_config.mutable_simulation(i)->
               mutable_aperture_params()->set_rotation(FLAGS_orientation);

    // Create the telescope
    vector<Mat> spectral_radiance;
    {
      Telescope telescope(sim_config, i, detector_params);
      CorrectExposure(image, wavelengths, illumination, telescope,
                      &spectral_radiance);
    }

    const double kRadialZoneWidth = 1 / max(1., FLAGS_radial_zones - 1.);
    const double kAngularZoneWidth = 2 * M_PI / FLAGS_angular_zones;

    vector<vector<Mat>> output_images;
    for (int r_zone = 0; r_zone < FLAGS_radial_zones; r_zone++) {
      for (int theta_zone = 0; theta_zone < FLAGS_angular_zones; theta_zone++) {
        double r = r_zone * kRadialZoneWidth;
        double theta = theta_zone * kAngularZoneWidth;

        coma_x->set_value(r * cos(theta) * FLAGS_peak_coma);
        coma_y->set_value(r * sin(theta) * FLAGS_peak_coma);

        cout << "Computing zone r " << r_zone + 1 << "/" << FLAGS_radial_zones
             << " theta " << theta_zone + 1 << "/" << FLAGS_angular_zones
             << StringPrintf(", Coma: (%.2f, %.2f)...",
                             coma_x->value(), coma_y->value()) << endl;

        // Image the input target
        Telescope telescope(sim_config, i, detector_params);
        output_images.emplace_back();
        vector<Mat> otf;
        telescope.Image(spectral_radiance, wavelengths,
                        &(output_images.back()), &otf);

        Mat psf;
        dft(otf[0], psf);
        imwrite(StringPrintf("sim_%d_zone_r_%d_theta_%d_psf.png",
                             i, r_zone, theta_zone),
                ColorScale(GammaScale(magnitude(FFTShift(psf)), 1/2.2),
                           COLORMAP_JET));
        if (r_zone == 0) {
          for (int i = 1; i < FLAGS_angular_zones; i++) {
            output_images.emplace_back(output_images.back());
          }
          theta_zone = FLAGS_angular_zones;
        }
      }
    }

    cout << "Combining..." << endl;
    vector<Mat> output_image;
    for (size_t band = 0; band < output_images[0].size(); band++) {
      output_image.emplace_back(output_images[0][0].size(), CV_64FC1);
      Mat& output = output_image.back();
      for (int i = 0; i < output.rows; i++) {
        double y = i - 0.5 * output.rows;
        for (int j = 0; j < output.cols; j++) {
          double x = j - 0.5 * output.cols;
          double r = sqrt(x*x + y*y) / (0.5 * max(output.rows, output.cols));
          double theta = atan2(y, x);
          while (theta < 0) theta += 2 * M_PI;

          double r_index = r / kRadialZoneWidth;
          double theta_index = theta / kAngularZoneWidth;

          int r_lt_index = min(max(int(floor(r_index)), 0),
                               FLAGS_radial_zones - 1);
          int r_gt_index = min(int(ceil(r_index)), FLAGS_radial_zones - 1);
          double r_blend = r_index - r_lt_index;
          int theta_lt_index = max(int(floor(theta_index)), 0);
          int theta_gt_index = int(ceil(theta_index)) % FLAGS_angular_zones;
          double theta_blend = theta_index - theta_lt_index;

          int lt_index = r_lt_index * FLAGS_angular_zones + theta_lt_index;
          int gt_index = r_gt_index * FLAGS_angular_zones + theta_lt_index;

          double r_lt_interp =
              (1 - r_blend) * output_images[lt_index][band].at<double>(i, j) +
              r_blend * output_images[gt_index][band].at<double>(i, j);

          lt_index = r_lt_index * FLAGS_angular_zones + theta_gt_index;
          gt_index = r_gt_index * FLAGS_angular_zones + theta_gt_index;
          double r_gt_interp =
              (1 - r_blend) * output_images[lt_index][band].at<double>(i, j) +
              r_blend * output_images[gt_index][band].at<double>(i, j);

          output.at<double>(i, j) = (1 - theta_blend) * r_lt_interp +
                                    theta_blend * r_gt_interp;
        }
      }
    }

    cout << "Restoring..." << endl;
    ConstrainedLeastSquares cls;

    coma_x->set_value(0);
    coma_y->set_value(0);
    Telescope telescope(sim_config, i, detector_params);

    // For each output band, perfom inverse filtering and write output
    for (size_t band = 0; band < output_image.size(); band++) {
      // Compute the spectral weighting for the effective OTF
      vector<double> spectral_weighting, det_qe;
      telescope.detector()->GetQESpectrum(wavelengths, band, &det_qe);
      for (size_t i = 0; i < wavelengths.size(); i++) {
        spectral_weighting.push_back(det_qe[i] * illumination[i]);
      }

      // Compute and output the effective OTF over the bandpass
      Mat deconvolved;
      Mat eff_otf;
      telescope.ComputeEffectiveOtf(wavelengths, spectral_weighting, &eff_otf);
      resize(eff_otf, eff_otf, output_image[band].size());
      imwrite(mats::StringPrintf("sim_%d_mtf_%d.png", i, band),
              GammaScale(FFTShift(magnitude(eff_otf)), 1/2.2));

      // Convert the raw image to the 0-1 range
      Mat raw_image;
      output_image[band].convertTo(raw_image, CV_64F,
          1 / pow(2., telescope.detector()->bit_depth()));

      // Deconvolve the image
      cls.Deconvolve(raw_image, eff_otf, FLAGS_smoothness, &deconvolved);

      // Contrast scale and output the raw and processed image
      raw_image.convertTo(raw_image, CV_8U, 255);
      imwrite(mats::StringPrintf("sim_%d_output_band_%d.png", i, band),
              raw_image);

      deconvolved.convertTo(deconvolved, CV_8U, 255);
      imwrite(mats::StringPrintf("sim_%d_processed_band_%d.png", i, band),
              deconvolved);

      // Output the inverse filter
      Mat inv_filter;
      cls.GetInverseFilter(eff_otf, FLAGS_smoothness, &inv_filter);
      imwrite(mats::StringPrintf("sim_%d_inv_filter_%d.png", i, band),
              GammaScale(FFTShift(magnitude(inv_filter)), 1/2.2));
    }
  }

  return 0;
}
