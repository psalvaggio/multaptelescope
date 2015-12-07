// Implementation file for global_circular_array.h
// Author: Philip Salvaggio

#include "global_circular_array.h"

namespace genetic {

auto GlobalCircularArray::Introduce(
    GeneticFitnessFunction<model_t>& fitness_function) -> model_t {
  model_t tmp_model;
  PopulationMember<model_t> member(std::move(tmp_model));
  model_t& locations(member.model());
  locations.resize(num_subapertures_);

  bool keep_going = true;
  int iter = 0;
  const int kWarningLevel = 1000;

  std::vector<double> subap_rs;
  for (const auto& tmp_r : subap_radii_) {
    for (int i = 0; i < tmp_r.second; i++) {
      subap_rs.push_back(tmp_r.first);
    }
  }

  while (keep_going) {
    iter++;
    std::random_shuffle(subap_rs.begin(), subap_rs.end());
    for (int i = 0; i < num_subapertures_; i++) {
      double max_center_radius = 0.5 * encircled_diameter_ - subap_rs[i];
      double r = (double)rand() / RAND_MAX * max_center_radius;
      double theta = (double)rand() / RAND_MAX * 2 * M_PI;
      locations[i].x = r * cos(theta);
      locations[i].y = r * sin(theta);
      locations[i].r = subap_rs[i];
    }
    keep_going = !fitness_function(member);
    if (iter % kWarningLevel == 0) {
      std::cerr << "WARNING: Introduce(): No valid model produced."
                << std::endl;
    }
  }

  model_t new_model = std::move(member.model());
  return new_model;
}


auto GlobalCircularArray::Crossover(
    const PopulationMember<model_t>& member1,
    const PopulationMember<model_t>& member2) -> model_t {
  const model_t& input1_locs(member1.model());
  const model_t& input2_locs(member2.model());
  model_t output_locs;
  output_locs.resize(num_subapertures_);

  int index = 0;
  for (size_t i = 0; i < subap_radii_.size(); i++) {
    // Detect the apertures from each operand for this radius
    std::array<std::vector<int>, 2> radii_indices;
    for (size_t j = 0; j < input1_locs.size(); j++) {
      if (input1_locs[j].r == subap_radii_[i].first) {
        radii_indices[0].push_back(j);
      }
      if (input2_locs[j].r == subap_radii_[i].first) {
        radii_indices[1].push_back(j);
      }
    }

    // Determine the spread of the sampling
    std::array<std::unordered_set<int>, 2> samples;
    for (int j = 0; j < subap_radii_[i].second; j++) {
       double p = (double)rand() / RAND_MAX;
       int operand_idx = (p < crossover_probability_) ? 0 : 1;
       size_t old_size = samples[operand_idx].size();
       do {
         samples[operand_idx].insert(rand() % subap_radii_[i].second);
       } while (samples[operand_idx].size() == old_size);
    }

    for (const auto& idx : samples[0])
      output_locs[index++] = input1_locs[radii_indices[0][idx]];
    for (const auto& idx : samples[1])
      output_locs[index++] = input2_locs[radii_indices[1][idx]];
  }

  return output_locs;
}


void GlobalCircularArray::Mutate(PopulationMember<model_t>& member) {
  model_t& locations(member.model());

  for (size_t i = 0; i < locations.size(); i++) {
    double p = (double)rand() / RAND_MAX;
    if (p < mutate_probability_) {
      double new_x = 2 * (double(rand()) / RAND_MAX - 0.5);
      double new_y = 2 * (double(rand()) / RAND_MAX - 0.5);
      double r2 = new_x * new_x + new_y * new_y;
      if (r2 > 1) {
        double r = sqrt(r2);
        new_x /= 1.0001 * r;
        new_y /= 1.0001 * r;
      }
      double max_center_radius = 0.5 * encircled_diameter_ - locations[i].r;
      new_x *= max_center_radius;
      new_y *= max_center_radius;

      locations[i].x = new_x;
      locations[i].y = new_y;
      
      p = (double)rand() / RAND_MAX;
      if (p < mutate_probability_) {
        int other = rand() % locations.size();
        double tmp = locations[i].r;
        locations[i].r = locations[other].r;
        locations[other].r = tmp;
      }
    }
  }
}

}  // namespace genetic
