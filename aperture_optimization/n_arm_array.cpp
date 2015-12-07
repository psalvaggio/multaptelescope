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

void NArmArray::SetNumArms(size_t arms) {
  apertures_.clear();
  arm_angles_.clear();
  arm_angles_.resize(arms);
  cache_dirty_ = true;
}


void NArmArray::SetArmAngle(int arm, double angle) {
  arm_angles_[arm] = angle;
  cache_dirty_ = true;
}


void NArmArray::AddAperture(int arm, double offset, double r) {
  apertures_.emplace_back(arm, offset, r);
  cache_dirty_ = true;
}


void NArmArray::clear() {
  arm_angles_.clear();
  apertures_.clear();
  cache_dirty_ = true;
}


void NArmArray::CreateCircularArray() const {
  circ_array_cache_.clear();
  for (const auto& ap : apertures_) {
    double arm_angle = arm_angles_[ap.arm];
    double from_center = ap.offset;
    circ_array_cache_.emplace_back(from_center * cos(arm_angle),
                                   from_center * sin(arm_angle),
                                   ap.r);
  }
  cache_dirty_ = false;
}


ostream& operator<<(ostream& os, const NArmArray& array) {
  size_t num_arms = array.NumArms();
  os << num_arms << endl;
  for (size_t i = 0; i < num_arms; i++) {
    os << array.ArmAngle(i) << endl;
    stringstream ss;
    size_t num_aps = 0;
    for (size_t j = 0; j < array.size(); j++) {
      if (array(j).arm == i) {
        ss << array(j).offset << " " << array(j).r << endl;
        num_aps++;
      }
    }
    os << num_aps << endl << ss.str();
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
