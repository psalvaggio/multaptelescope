// File Description
// Author: Philip Salvaggio

#include "golay_fitness_function.h"

#include "base/kd_tree.h"
#include "base/pupil_function.h"
#include "base/opencv_utils.h"
#include "optical_designs/compound_aperture.h"

#include <array>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace mats;

namespace genetic {

GolayFitnessFunction::GolayFitnessFunction(int num_subapertures,
                                           double encircled_diameter,
                                           double subaperture_diameter)
    : max_center_radius2_(0),
      subaperture_diameter2_(subaperture_diameter*subaperture_diameter),
      encircled_diameter_(encircled_diameter),
      peaks_(2 * (num_subapertures * (num_subapertures - 1) + 1), 0) {
  max_center_radius2_ = pow(0.5*(encircled_diameter-subaperture_diameter), 2);
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

  // Calculate the compactness of the MTF
  GetAutocorrelationPeaks(member.model(), &peaks_);
  for (size_t i = 0; i < peaks_.size(); i += 2) {
    moment_of_inertia += (peaks_[i]*peaks_[i] + peaks_[i+1]*peaks_[i+1]) / 
                         max_center_radius2_;
  }
  double compactness = 1 / moment_of_inertia;

  KDTree<array<double, 2>, 2> kd_tree;
  for (size_t i = 0; i < peaks_.size(); i += 2) {
    kd_tree.emplace_back();
    kd_tree.back() = {{peaks_[i], peaks_[i+1]}};
  }
  kd_tree.build();
  
  const int kSamples = 75;
  const double kAutocorWidth = 2 * encircled_diameter_;
  const double kPeakWidth = sqrt(subaperture_diameter2_);
  const double kPeakHeight = 1 / 6.;
  array<double, 2> sample;
  vector<int> neighbors;
  int covered = 0;
  for (int i = 0; i < kSamples; i++) {
    sample[1] = kAutocorWidth * (i - kSamples / 2.) / kSamples;
    for (int j = 0; j < kSamples; j++) {
      sample[0] = kAutocorWidth * (j - kSamples / 2.) / kSamples;

      kd_tree.kNNSearch(sample, 1, kPeakWidth, &neighbors);
      if (!neighbors.empty()) {
        double approx_mtf = 0;
        for (size_t k = 0; k < neighbors.size(); k++) {
          double dist = sqrt(pow(kd_tree[neighbors[k]][0] - sample[0], 2) +
                             pow(kd_tree[neighbors[k]][1] - sample[1], 2));
          approx_mtf += kPeakHeight * (1 - dist / kPeakWidth);
        }
        if (approx_mtf > 0.03) covered++;
      }
    }
  }
  double support_frac = covered / (M_PI * pow(kSamples / 2., 2));

  double fitness = support_frac + 5 * compactness;
  member.set_fitness(fitness);

  return true;
}


void GolayFitnessFunction::Visualize(const model_t& locations) {
  Simulation sim;
  auto* compound_params = sim.mutable_aperture_params();
  compound_params->set_encircled_diameter(encircled_diameter_);
  compound_params->set_type(ApertureParameters::COMPOUND);

  auto* array_ext = compound_params->MutableExtension(compound_aperture_params);
  array_ext->set_combine_operation(CompoundApertureParameters::OR);
  for (size_t i = 0; i < locations.size() / 2; i++) {
    auto* subap = array_ext->add_aperture();
    subap->set_type(ApertureParameters::CIRCULAR);
    subap->set_encircled_diameter(sqrt(subaperture_diameter2_));
    subap->set_offset_x(locations[2*i]);
    subap->set_offset_y(locations[2*i+1]);
  }

  PupilFunction pupil(512, 550e-9);
  unique_ptr<Aperture> aperture(ApertureFactory::Create(sim));
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
  int subaps = locations.size() / 2;
  peaks->resize(2 * (subaps * (subaps - 1) + 1), 0);

  int index = 2;
  for (size_t i = 0; i < locations.size(); i += 2) {
    for (size_t j = 0; j < locations.size(); j += 2) {
      if (i == j) continue;

      (*peaks)[index++] = locations[i] - locations[j];
      (*peaks)[index++] = locations[i+1] - locations[j+1];
    }
  }
}

}  // namespace genetic
