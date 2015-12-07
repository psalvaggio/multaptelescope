// Implementation file for global_n_arm.h
// Author: Philip Salvaggio

#include "global_n_arm.h"

using namespace std;

namespace genetic {

auto GlobalNArm::Introduce(
    GeneticFitnessFunction<model_t>& fitness_function) -> model_t {
  model_t tmp_model;
  PopulationMember<model_t> member(std::move(tmp_model));
  model_t& array = member.model();
  const double kEncircledR = 0.5 * encircled_diameter_;

  // Keep generating apertures until we get a valid one.
  bool keep_going = true;
  while (keep_going) {
    array.clear();
    array.SetNumArms(num_arms_);

    // Construct random offset/radii pairs.
    vector<pair<double, double>> ap_offset_rs;
    for (const auto& tmp_r : subap_radii_) {
      for (int i = 0; i < tmp_r.second; i++) {
        double max_r = kEncircledR - 2 * tmp_r.first;
        double offset = max_r * (double(rand()) / RAND_MAX) + tmp_r.first;
        ap_offset_rs.emplace_back(offset, tmp_r.first);
      }
    }

    // Shuffle indices into ap_offset_rs
    vector<int> ap_indices(ap_offset_rs.size());
    iota(begin(ap_indices), end(ap_indices), 0);
    random_shuffle(begin(ap_indices), end(ap_indices));

    // Randomly assign apertures to randomly created arms
    int floating_aps = ap_offset_rs.size() - num_arms_;
    int ap_index = 0;
    for (int i = 0; i < num_arms_; i++) {
      array.SetArmAngle(i, 2 * M_PI * (double(rand()) / RAND_MAX));

      int num_aps = rand() % (floating_aps + 1);
      if (i == num_arms_ - 1) num_aps = floating_aps;

      floating_aps -= num_aps;
      num_aps++;
      for (int j = 0; j < num_aps; j++) {
        const auto& ap_params = ap_offset_rs[ap_indices[ap_index++]];
        array.AddAperture(i, ap_params.first, ap_params.second);
      }
    }
    array.SetArmAngle(0, 0);  // Fix the first arm always at 0.

    keep_going = !fitness_function(member);
  }

  model_t new_model = move(member.model());
  return new_model;
}


auto GlobalNArm::Crossover(
    const PopulationMember<model_t>& member1,
    const PopulationMember<model_t>& member2) -> model_t {
  const model_t& array1 = member1.model();
  const model_t& array2 = member2.model();

  // Set up the arms by randomly combining the arms from the two operands.
  model_t result;
  result.SetNumArms(num_arms_);
  result.SetArmAngle(0, 0);
  for (int i = 1; i < num_arms_; i++) {
    double p = double(rand()) / RAND_MAX;
    const model_t& src = p < crossover_probability_ ? array2 : array1;
    result.SetArmAngle(i, src.ArmAngle(i));
  }

  // We need to respect the subaperture radius budget, so we'll add the correct
  // number of apertures for each radius.
  const size_t kNumAps = array1.size();
  for (size_t i = 0; i < subap_radii_.size(); i++) {
    // Locate the subapertures in the operands that have the current radius.
    array<vector<int>, 2> radii_indices;
    for (size_t j = 0; j < kNumAps; j++) {
      if (array1(j).r == subap_radii_[i].first) {
        radii_indices[0].emplace_back(j);
      }
      if (array2(j).r == subap_radii_[i].first) {
        radii_indices[1].emplace_back(j);
      }
    }

    // Determine the spread of the sampling
    array<unordered_set<int>, 2> samples;
    for (int j = 0; j < subap_radii_[i].second; j++) {
      double p = (double)rand() / RAND_MAX;
      int operand_idx = (p < crossover_probability_) ? 1 : 0;
      size_t old_size = samples[operand_idx].size();
      do {
        samples[operand_idx].insert(rand() % subap_radii_[i].second);
      } while (samples[operand_idx].size() == old_size);
    }

    for (int j = 0; j < 2; j++) {
      const auto& src = j == 0 ? array1 : array2;
      for (auto sample : samples[j]) {
        const auto& ap = src(radii_indices[j][sample]);
        int arm = rand() % num_arms_;
        result.AddAperture(arm, ap.offset, ap.r);
      }
    }
  }

  return result;
}


void GlobalNArm::Mutate(PopulationMember<model_t>& member) {
  model_t& array = member.model();

  // Randomly rotate arms
  for (int i = 1; i < num_arms_; i++) {
    double p = double(rand()) / RAND_MAX;
    if (p < mutate_probability_) {
      array.SetArmAngle(i, 2 * M_PI * (double(rand()) / RAND_MAX));
    }
  }

  // Randomly change offsets or swap radii
  const int kNumAps = array.size();
  for (int i = 0; i < kNumAps; i++) {
    double p = double(rand()) / RAND_MAX;
    if (p >= mutate_probability_) continue;

    auto& ap = array(i);

    p = double(rand()) / RAND_MAX;
    if (p < 0.5) {
      ap.arm = p < 0.25 ? rand() % num_arms_ : ap.arm;
      double max_r = 0.5 * encircled_diameter_ - 2 * ap.r;
      ap.offset = max_r * (double(rand()) / RAND_MAX) + ap.r;
    } else {
      swap(ap.r, array(rand() % kNumAps).r);
    }
  }
}

}  // namespace genetic
