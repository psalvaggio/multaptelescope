// File Description
// Author: Philip Salvaggio

#ifndef ANNULUS_FITNESS_FUNCTION_H
#define ANNULUS_FITNESS_FUNCTION_H

#include "annulus_fitness_function.h"

#include "base/kd_tree.h"
#include "base/pupil_function.h"
#include "base/opencv_utils.h"
#include "base/str_utils.h"
#include "optical_designs/compound_aperture.h"

#include <array>

#include <opencv2/highgui/highgui.hpp>

namespace genetic {

template<typename T>
const int AnnulusFitnessFunction<T>::kMtfSize = 75;

template<typename T>
AnnulusFitnessFunction<T>::AnnulusFitnessFunction(
    int num_subapertures,
    double encircled_diameter,
    const CircularSubapertureBudget& subaperture_radii)
    : subap_radii_(subaperture_radii),
      encircled_diameter_(encircled_diameter),
      peaks_(num_subapertures * (num_subapertures - 1) + 1),
      max_subap_radius_(0),
      total_r2_(0),
      radial_masks_(),
      mask_radii_() {
  for (const auto& subap_r : subap_radii_) {
    if (subap_r.second > 0) {
      max_subap_radius_ = std::max(max_subap_radius_, subap_r.first);
    }
    total_r2_ += subap_r.second * subap_r.first * subap_r.first;
  }

  int big_r = 0.95 * kMtfSize / 2;
  int little_r = 0.6 * kMtfSize / 2;
  int r_step = 0.05 * kMtfSize / 2;
  for (int r = little_r; r <= big_r; r += r_step) {
    mask_radii_.emplace_back(r);
    radial_masks_.emplace_back(kMtfSize, kMtfSize);
    radial_masks_.back() = 0;
    cv::circle(radial_masks_.back(), cv::Point(kMtfSize / 2, kMtfSize / 2), r,
               cv::Scalar(1, 0, 0), -1);
  }
}


template<typename T>
bool AnnulusFitnessFunction<T>::operator()(
    PopulationMember<model_t>& member) const {
  const model_t& locations(member.model());

  // Check for non-overlapping subapertures
  for (size_t i = 0; i < locations.size(); i++) {
    const auto& ap = locations[i];
    double dist2 = ap.x * ap.x + ap.y * ap.y;
    double dist = sqrt(dist2);
    dist += ap.r;
    if (dist > 0.5*encircled_diameter_) return false;

    for (size_t j = i + 1; j < locations.size(); j++) {
      const auto& other = locations[j];
      double dx = ap.x - other.x;
      double dy = ap.y - other.y;
      double distance2 = dx*dx + dy*dy;
      double min_dist = ap.r + other.r;
      if (distance2 < min_dist * min_dist) return false;
    }
  }

  // Build a KD-tree for auto-correlation peaks.
  KDTree<std::array<double, 2>, 2> kd_tree;
  GetAutocorrelationPeaks(member.model(), &peaks_);
  for (const auto& peak : peaks_) {
    kd_tree.emplace_back();
    kd_tree.back() = {{peak.x, peak.y}};
  }
  kd_tree.build();

  // For "isotropic-ness" and smootheness, we'll be looking at the distribution
  // of MTF support as a function of radius. For an annulus design, encircled
  // MTF support is almost perfectly linearly related to radius between 25% and
  // 75% of the MTF energy. So, we'll be maximizing the correlation coefficient
  // over this range.
  std::vector<std::pair<double, double>> mtf_distribution;
  for (int i = 0; i < kMtfSize / 2; i++) {
    mtf_distribution.emplace_back(i, 0);
  }
  double total_mtf = 0;
  
  const double kAutocorWidth = 2 * encircled_diameter_;
  const double kMaxPeakRadius = 2 * max_subap_radius_;
  std::array<double, 2> sample;
  std::vector<int> neighbors;
  int covered = 0;
  for (int i = 0; i < kMtfSize; i++) {
    sample[1] = kAutocorWidth * (i - kMtfSize / 2.) / kMtfSize;
    for (int j = 0; j < kMtfSize; j++) {
      sample[0] = kAutocorWidth * (j - kMtfSize / 2.) / kMtfSize;

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

        total_mtf += approx_mtf;
        double r = sqrt(sample[0] * sample[0] + sample[1] * sample[1]);
        int index = (kMtfSize / 2.) * r / encircled_diameter_;

        if (index >= 0 && index < int(mtf_distribution.size())) {
          mtf_distribution[index].second += approx_mtf;
        }
      }
    }
  }
  double support_frac = covered / (M_PI * pow(kMtfSize / 2., 2));

  // Construct the MTF distribution as a function of radius.
  mtf_distribution[0].second /= total_mtf;
  for (size_t i = 1; i < mtf_distribution.size(); i++) {
    mtf_distribution[i].second /= total_mtf;
    mtf_distribution[i].second += mtf_distribution[i-1].second;
  }

  
  const double kMtfPercentAreaStart = 0.1;
  const double kMtfPercentAreaEnd = 0.75;
  int start_idx = 0;
  int end_idx = mtf_distribution.size() - 1;
  while (mtf_distribution[start_idx].second < kMtfPercentAreaStart &&
         start_idx < int(mtf_distribution.size())) {
    start_idx++;
  }
  while (mtf_distribution[end_idx].second > kMtfPercentAreaEnd &&
         end_idx > 0) {
   end_idx--;
  }

  double sum_r_mtf = 0, sum_r_sq = 0, sum_r = 0, sum_mtf = 0, sum_mtf_sq = 0;
  for (int i = start_idx; i <= end_idx; i++) {
    double r = mtf_distribution[i].first;
    double mtf = mtf_distribution[i].second;
    sum_r_mtf += r * mtf;
    sum_r_sq += r * r;
    sum_r += r;
    sum_mtf_sq += mtf * mtf;
    sum_mtf += mtf;
  }
  int size = end_idx - start_idx + 1;
  sum_r_mtf /= size;
  sum_r_sq /= size;
  sum_r /= size;
  sum_mtf /= size;
  sum_mtf_sq /= size;

  double correlation = (sum_r_mtf - sum_r * sum_mtf) /
      (sqrt(sum_r_sq - sum_r * sum_r) * sqrt(sum_mtf_sq - sum_mtf * sum_mtf));
  double corr_metric = std::max(0., 50 * (correlation - 0.98));

  double fitness = corr_metric + 0.5 * support_frac;
  member.set_fitness(fitness);

  return true;
}


template<typename T>
void AnnulusFitnessFunction<T>::Visualize(const model_t& locations) const {
  mats::Simulation sim;
  auto* compound_params = sim.mutable_aperture_params();
  compound_params->set_encircled_diameter(encircled_diameter_);
  compound_params->set_type(mats::ApertureParameters::COMPOUND);

  auto* array_ext = compound_params->MutableExtension(compound_aperture_params);
  array_ext->set_combine_operation(CompoundApertureParameters::OR);
  for (size_t i = 0; i < locations.size(); i++) {
    const auto& ap = locations[i];
    auto* subap = array_ext->add_aperture();
    subap->set_type(mats::ApertureParameters::CIRCULAR);
    subap->set_encircled_diameter(2 * ap.r);
    subap->set_offset_x(ap.x);
    subap->set_offset_y(ap.y);
  }

  mats::PupilFunction pupil(512, 550e-9);
  std::unique_ptr<Aperture> aperture(ApertureFactory::Create(sim));
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
void AnnulusFitnessFunction<T>::GetAutocorrelationPeaks(
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
