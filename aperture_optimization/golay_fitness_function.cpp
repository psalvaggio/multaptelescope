// File Description
// Author: Philip Salvaggio

#include "golay_fitness_function.h"

#include "base/pupil_function.h"
#include "base/opencv_utils.h"
#include "optical_designs/compound_aperture.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace mats;

namespace genetic {

GolayFitnessFunction::GolayFitnessFunction(int num_subapertures,
                                           double encircled_diameter,
                                           double subaperture_diameter,
                                           double reference_wavelength)
    : conf_(),
      max_center_radius2_(0),
      subaperture_diameter2_(subaperture_diameter*subaperture_diameter) {

  // Set up the simulation parameters, so we can just change the subaperture 
  // locations later.
  max_center_radius2_ = pow(0.5*(encircled_diameter-subaperture_diameter), 2);
  conf_.set_array_size(512);
  conf_.set_reference_wavelength(reference_wavelength);

  Simulation* sim = conf_.add_simulation();
  auto* compound_params = sim->mutable_aperture_params();
  compound_params->set_encircled_diameter(encircled_diameter);

  compound_params->set_type(ApertureParameters::COMPOUND);
  auto* array_ext = compound_params->MutableExtension(compound_aperture_params);
  array_ext->set_combine_operation(CompoundApertureParameters::OR);
  for (int i = 0; i < num_subapertures; i++) {
    auto* subap = array_ext->add_aperture();
    subap->set_type(ApertureParameters::CIRCULAR);
    subap->set_encircled_diameter(subaperture_diameter);
    subap->set_offset_x(0);
    subap->set_offset_y(0);
  }
}

bool GolayFitnessFunction::operator()(PopulationMember<model_t>& member) {
  double moment_of_inertia = 0;

  model_t& locations(member.model());

  // Check for non-overlapping subapertures
  for (size_t i = 0; i < locations.size(); i += 2) {
    double sq_dist = locations[i]*locations[i] + locations[i+1]*locations[i+1];
    if (sq_dist > max_center_radius2_) return false;

    for (size_t j = i + 2; j < locations.size(); j += 2) {
      double dx = locations[i] - locations[j];
      double dy = locations[i+1] - locations[j+1];
      double distance2 = dx*dx + dy*dy;
      if (distance2 < subaperture_diameter2_) return false;
    }
  }

  // Set the subaperture positions in our parameters
  auto* ap_params = conf_.mutable_simulation(0)->mutable_aperture_params();
  auto* compound_params = ap_params->MutableExtension(compound_aperture_params);
  for (size_t i = 0; i < locations.size(); i += 2) {
    auto* subap = compound_params->mutable_aperture(i/2);
    subap->set_offset_x(locations[i]);
    subap->set_offset_y(locations[i+1]);
  }

  // Create the proposed aperture
  unique_ptr<Aperture> aperture(ApertureFactory::Create(conf_.simulation(0)));

  // Calculate the monochromatic, on-axis MTF
  PupilFunction pupil(conf_.array_size(), conf_.reference_wavelength());
  aperture->GetPupilFunction(conf_.reference_wavelength(), 0, 0, &pupil);
  cv::Mat mtf = pupil.ModulationTransferFunction();

  // Calculate the non-redencance of the MTF (as a percentage)
  cv::Mat mtf_float;
  mtf.convertTo(mtf_float, CV_32F);
  cv::threshold(mtf_float, mtf_float, 0.03, 1, cv::THRESH_TOZERO);
  int non_zeros = cv::countNonZero(mtf_float);
  double support_frac = non_zeros / (M_PI * pow(mtf.cols / 2, 2));
  
  // Calculate the compactness of the MTF
  model_t peaks;
  GetAutocorrelationPeaks(member.model(), &peaks);
  for (size_t i = 0; i < peaks.size(); i += 2) {
    moment_of_inertia += (peaks[i]*peaks[i] + peaks[i+1]*peaks[i+1]) / 
                         max_center_radius2_;
  }
  double compactness = 1 / moment_of_inertia;

  double fitness = support_frac + 1 * compactness;
  member.set_fitness(fitness);

  return true;
}

void GolayFitnessFunction::Visualize(const model_t& locations) {
  ApertureParameters* ap_params(
      conf_.mutable_simulation(0)->mutable_aperture_params());
  CompoundApertureParameters* compound_params =
      ap_params->MutableExtension(compound_aperture_params);
  for (size_t i = 0; i < locations.size(); i += 2) {
    ApertureParameters* subap = compound_params->mutable_aperture(i/2);
    subap->set_offset_x(locations[i]);
    subap->set_offset_y(locations[i+1]);
  }

  PupilFunction pupil(conf_.array_size(), conf_.reference_wavelength());
  unique_ptr<Aperture> aperture(ApertureFactory::Create(conf_.simulation(0)));
  aperture->GetPupilFunction(550e-9, 0, 0, &pupil);
  cv::Mat mtf = pupil.ModulationTransferFunction();
  cv::Mat mask = ByteScale(aperture->GetApertureMask(512));
  cv::circle(mask, cv::Point(mask.cols / 2, mask.rows / 2), mask.cols / 2,
             cv::Scalar(255, 255, 0), 2);
  cv::imshow("Best MTF", FFTShift(GammaScale(mtf, 1/2.2)));
  cv::imshow("Best Mask", mask);
  cv::waitKey(1);
}


void GolayFitnessFunction::GetAutocorrelationPeaks(const model_t& locations,
                                                   model_t* peaks) {
  if (!peaks) return;
  peaks->resize(locations.size() * (locations.size() - 1) + 1, 0);

  int index = 1;
  for (size_t i = 0; i < locations.size(); i += 2) {
    for (size_t j = 0; j < locations.size(); j += 2) {
      if (i == j) continue;

      (*peaks)[index++] = locations[i] - locations[j];
      (*peaks)[index++] = locations[i+1] - locations[j+1];
    }
  }
}

}  // namespace genetic
