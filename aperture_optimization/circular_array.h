// A parameterization of sparse apertures as an array of circular apertures
// Author: Philip Salvaggio

#ifndef CIRCULAR_ARRAY_H
#define CIRCULAR_ARRAY_H

#include "base/simulation_config.pb.h"

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>

namespace genetic {

class CircularSubaperture {
 public:
  double x, y, r;

 public:
  CircularSubaperture() : x(0), y(0), r(0) {}
  CircularSubaperture(double x, double y, double r) : x(x), y(y), r(r) {}

  CircularSubaperture& operator=(const CircularSubaperture& other) = default;

  friend std::ostream& operator<<(std::ostream &output,
                                  const CircularSubaperture& self);
};

using CircularSubapertureBudget = std::vector<std::pair<double, int>>;

inline void CircularSubapBudgetHelper(CircularSubapertureBudget&) {}

template<typename... Args>
inline void CircularSubapBudgetHelper(CircularSubapertureBudget& subaps,
    double radius, int count, Args&&... args) {
  subaps.emplace_back(radius, count);
  CircularSubapBudgetHelper(subaps, std::forward<Args>(args)...);
}

template<typename... Args>
inline CircularSubapertureBudget MakeCircularSubapertureBudget(Args&&... args) {
  CircularSubapertureBudget budget;
  CircularSubapBudgetHelper(budget, std::forward<Args>(args)...);
  return budget;
}

class CircularAutocorrelationPeak {
 public:
  double x, y, height, min_r, max_r;

 public:
  CircularAutocorrelationPeak()
      : x(0), y(0), height(0), min_r(0), max_r(0) {}

  CircularAutocorrelationPeak(
      const CircularAutocorrelationPeak& other) = default;
  CircularAutocorrelationPeak& operator=(
      const CircularAutocorrelationPeak& other) = default;

  void set(double x, double y, double height, double min_r, double max_r) {
    this->x = x; this->y = y; this->height = height; this->min_r = min_r;
    this->max_r = max_r;
  }

  double& operator[](int index) { return (index == 0) ? x : y; }
  const double& operator[](int index) const { return (index == 0) ? x : y; }
};


using CircularArray = std::vector<CircularSubaperture>;
std::ostream& operator<<(std::ostream& os, const CircularArray& aps);
std::istream& operator>>(std::istream& is, CircularArray& aps);

}  // namespace genetic

#endif  // CIRCULAR_ARRAY_H
