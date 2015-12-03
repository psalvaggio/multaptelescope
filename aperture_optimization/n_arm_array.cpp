// A parameterization of sparse apertures as a number of arms, which contain
// collinear circular apertures.
// Author: Philip Salvaggio

#include "n_arm_array.h"

using namespace std;

namespace genetic {

const CircularSubaperture& NArmArray::operator[](int index) const {
  if (cache_dirty_) CreateCircularArray();

  return circ_array_cache_[index];
}

auto NArmArray::operator()(size_t index) const -> const ArmSubaperture& {
  int arm = 0;
  while (index >= apertures_[arm].size()) {
    index -= apertures_[arm].size();
    arm++;
  }
  return apertures_[arm][index];
}


void NArmArray::SetNumArms(size_t arms) {
  arm_angles_.resize(arms);
  apertures_.resize(arms);

  size_ = 0;
  for (const auto& arm : apertures_) size_ += arm.size();

  cache_dirty_ = true;
}


void NArmArray::SetArmAngle(int arm, double angle) {
  arm_angles_[arm] = angle;
  cache_dirty_ = true;
}


void NArmArray::AddAperture(int arm, double offset, double r) {
  apertures_[arm].emplace_back(offset, r);
  cache_dirty_ = true;
  size_++;
}


void NArmArray::RemoveAperture(int arm, int ap) {
  apertures_[arm].erase(begin(apertures_[arm]) + ap);
}


void NArmArray::clear() {
  arm_angles_.clear();
  apertures_.clear();
  size_ = 0;
  cache_dirty_ = true;
}


void NArmArray::CreateCircularArray() const {
  circ_array_cache_.clear();
  for (size_t i = 0; i < apertures_.size(); i++) {
    double arm_angle = arm_angles_[i];
    for (size_t j = 0; j < apertures_[i].size(); j++) {
      double from_center = apertures_[i][j].offset;
      double subap_r = apertures_[i][j].r;
      circ_array_cache_.emplace_back(from_center * cos(arm_angle),
                                     from_center * sin(arm_angle),
                                     subap_r);
    }
  }
  cache_dirty_ = false;
}


ostream& operator<<(ostream& os, const NArmArray& array) {
  size_t num_arms = array.NumArms();
  os << num_arms << endl;
  for (size_t i = 0; i < num_arms; i++) {
    os << array.ArmAngle(i) << endl;
    size_t num_aps = array.AperturesOnArm(i);
    os << num_aps << endl;
    for (size_t j = 0; j < num_aps; j++) {
     os << array(i, j).offset << " " << array(i, j).r << endl;
    }
  }
  return os;
}

istream& operator>>(istream& is, NArmArray& array) {
  size_t num_arms = 0;
  string line;
  istringstream iss;
  if (getline(is, line)) {
    iss.str(line);
    iss >> num_arms;
  }
  array.SetNumArms(num_arms);

  for (size_t i = 0; i < num_arms; i++) {
    double angle = 0;
    if (getline(is, line)) {
      iss.clear();
      iss.str(line);
      iss >> angle;
      array.SetArmAngle(i, angle);
    }

    size_t num_aps = 0;
    if (getline(is, line)) {
      iss.clear();
      iss.str(line);
      iss >> num_aps;
    }
    for (size_t j = 0; j < num_aps; j++) {
      if (getline(is, line)) {
        iss.clear();
        iss.str(line);
        double offset, r;
        iss >> offset >> r;
        array.AddAperture(i, offset, r);
      }
    }
  }
  return is;
}

}  // namespace genetic
