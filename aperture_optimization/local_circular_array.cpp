// Implementation file for local_circular_array.h
// Author: Philip Salvaggio

#include "local_circular_array.h"

namespace genetic {

void LocalCircularArray::Mutate(PopulationMember<model_t>& member) {
  model_t& locations(member.model());

  const double kEncircledRadius = encircled_diameter_ / 2.0;

  for (size_t i = 0; i < locations.size(); i++) {
    double p = (double)rand() / RAND_MAX;
    if (p < mutate_probability_) {
      locations[i].x += distribution_(generator_) * subap_translate_stddev_;
      locations[i].y += distribution_(generator_) * subap_translate_stddev_;
      double r = sqrt(pow(locations[i].x, 2) + pow(locations[i].y, 2));
      if (r > kEncircledRadius - locations[i].r) {
        double scale = 0.999 * (kEncircledRadius - locations[i].r) / r;
        locations[i].x *= scale;
        locations[i].y *= scale;
      }
      if ((double)rand() / RAND_MAX < 0.01) {
        double new_x = 2 * (double(rand()) / RAND_MAX - 0.5);
        double new_y = 2 * (double(rand()) / RAND_MAX - 0.5);
        double r2 = new_x * new_x + new_y * new_y;
        if (r2 > 1) {
          double r = sqrt(r2);
          new_x /= r;
          new_y /= r;
        }
        locations[i].x = (kEncircledRadius - locations[i].r) * new_x;
        locations[i].y = (kEncircledRadius - locations[i].r) * new_y;
      }
    }
  }
}

}  // namespace genetic
