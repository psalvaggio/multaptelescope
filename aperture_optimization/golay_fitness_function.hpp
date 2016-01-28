// File Description
// Author: Philip Salvaggio

#ifndef GOLAY_FITNESS_FUNCTION_HPP
#define GOLAY_FITNESS_FUNCTION_HPP

#include "golay_fitness_function.h"

#include "base/kd_tree.h"
#include "base/pupil_function.h"
#include "base/opencv_utils.h"
#include "optical_designs/compound_aperture.h"

#include <array>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace genetic {

template<typename T>
GolayFitnessFunction<T>::GolayFitnessFunction(
    int num_subapertures,
    double encircled_diameter,
    const CircularSubapertureBudget& subaperture_radii)
    : subap_radii_(subaperture_radii),
      encircled_diameter_(encircled_diameter),
      peaks_(num_subapertures * (num_subapertures - 1) + 1),
      max_subap_radius_(0),
      total_r2_(0) {
  for (const auto& subap_r : subap_radii_) {
    if (subap_r.second > 0) {
      max_subap_radius_ = std::max(max_subap_radius_, subap_r.first);
    }
    total_r2_ += subap_r.second * subap_r.first * subap_r.first;
  }
}


template<typename T>
bool GolayFitnessFunction<T>::operator()(
    PopulationMember<model_t>& member) const {
  double moment_of_inertia = 0;

  const model_t& locations(member.model());

  // Check for non-overlapping subapertures
  for (size_t i = 0; i < locations.size(); i++) {
    double dist2 = locations[i].x * locations[i].x +
                   locations[i].y * locations[i].y;
    double dist = sqrt(dist2);
    dist += locations[i].r;
    if (dist > 0.5*encircled_diameter_) return false;

    for (size_t j = i + 1; j < locations.size(); j++) {
      double dx = locations[i].x - locations[j].x;
      double dy = locations[i].y - locations[j].y;
      double distance2 = dx*dx + dy*dy;
      double min_dist = locations[i].r + locations[j].r;
      if (distance2 < min_dist * min_dist) return false;
    }
  }

  // Calculate the compactness of the MTF
  double norm = 0.25 * encircled_diameter_ * encircled_diameter_;
  GetAutocorrelationPeaks(member.model(), &peaks_);
  for (const auto& peak : peaks_) {
    moment_of_inertia += (peak.x * peak.x + peak.y * peak.y) / norm;
  }
  double compactness = 1 / moment_of_inertia;

  KDTree<std::array<double, 2>, 2> kd_tree;
  for (const auto& peak : peaks_) {
    kd_tree.emplace_back();
    kd_tree.back() = {{peak.x, peak.y}};
  }
  kd_tree.build();
  
  const int kSamples = 75;
  const double kAutocorWidth = 2 * encircled_diameter_;
  const double kMaxPeakRadius = 2 * max_subap_radius_;
  std::array<double, 2> sample;
  std::vector<int> neighbors;
  int covered = 0;
  for (int i = 0; i < kSamples; i++) {
    sample[1] = kAutocorWidth * (i - kSamples / 2.) / kSamples;
    for (int j = 0; j < kSamples; j++) {
      sample[0] = kAutocorWidth * (j - kSamples / 2.) / kSamples;

      kd_tree.kNNSearch(sample, -1, kMaxPeakRadius, &neighbors);
      if (!neighbors.empty()) {
        double approx_mtf = 0;
        for (size_t k = 0; k < neighbors.size(); k++) {
          const auto& peak = peaks_[neighbors[k]];
          double dist = sqrt(pow(peak.x - sample[0], 2) +
                             pow(peak.y - sample[1], 2)) - peak.min_r;
          if (dist < 0) {
            approx_mtf += peak.height;
          } else if (dist < peak.max_r - peak.min_r) {
            approx_mtf += peak.height * (1 - dist / (peak.max_r - peak.min_r));
          }
        }
        if (approx_mtf > 0.03) covered++;
      }
    }
  }
  double support_frac = covered / (M_PI * pow(kSamples / 2., 2));

  double fitness = support_frac + 3 * compactness;
  member.set_fitness(fitness);

  return true;
}


template<typename T>
void GolayFitnessFunction<T>::Visualize(const model_t& locations) const {
  mats::Simulation sim;
  auto* compound_params = sim.mutable_aperture_params();
  compound_params->set_encircled_diameter(encircled_diameter_);
  compound_params->set_type(mats::ApertureParameters::COMPOUND);

  auto* array_ext = compound_params->MutableExtension(compound_aperture_params);
  array_ext->set_combine_operation(CompoundApertureParameters::OR);
  for (size_t i = 0; i < locations.size(); i++) {
    auto* subap = array_ext->add_aperture();
    subap->set_type(mats::ApertureParameters::CIRCULAR);
    subap->set_encircled_diameter(2 * locations[i].r);
    subap->set_offset_x(locations[i].x);
    subap->set_offset_y(locations[i].y);
  }

  mats::PupilFunction pupil(512, 550e-9);
  std::unique_ptr<mats::Aperture> aperture(mats::ApertureFactory::Create(sim));
  aperture->GetPupilFunction(550e-9, 0, 0, &pupil);

  cv::Mat mtf = pupil.ModulationTransferFunction();
  cv::Mat mask = ByteScale(aperture->GetApertureMask(512));
  cv::circle(mask, cv::Point(mask.cols / 2, mask.rows / 2), mask.cols / 2,
             cv::Scalar(255, 255, 0), 2);
  cv::imshow("Best MTF", FFTShift(GammaScale(mtf, 1/2.2)));
  cv::imshow("Best Mask", mask);
  cv::waitKey(1);
}


template<typename T>
void GolayFitnessFunction<T>::GetAutocorrelationPeaks(
    const model_t& locations,
    std::vector<CircularAutocorrelationPeak>* peaks) const {
  if (!peaks) return;
  int subaps = locations.size();
  peaks->resize((subaps * (subaps - 1) + 1));

  (*peaks)[0].set(0, 0, 1, 0, max_subap_radius_ * 2);

  int index = 1;
  for (size_t i = 0; i < locations.size(); i++) {
    for (size_t j = 0; j < locations.size(); j++) {
      if (i == j) continue;
      double smaller_r = std::min(locations[i].r, locations[j].r);
      double bigger_r = std::max(locations[i].r, locations[j].r);

      (*peaks)[index++].set(locations[i].x - locations[j].x,
                            locations[i].y - locations[j].y,
                            pow(smaller_r, 2) / total_r2_,
                            bigger_r - smaller_r,
                            bigger_r + smaller_r);
    }
  }
}

}  // namespace genetic

#endif
