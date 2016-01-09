// A parameterization of sparse apertures as an array of circular apertures
// Author: Philip Salvaggio

#include "circular_array.h"

using namespace std;

namespace genetic {

ostream& operator<<(ostream &output, const CircularSubaperture& self) { 
  output << "Subaperture: X=" << self.x << ", Y=" << self.y << ", R="
         << self.r;
  return output;            
}

ostream& operator<<(ostream& os, const CircularArray& aps) {
  os << aps.size() << endl;
  for (const auto& ap : aps) {
    os << ap.x << " " << ap.y << " " << ap.r << endl;
  }
  return os;
}

istream& operator>>(istream& is, CircularArray& aps) {
  string line;
  istringstream iss;

  int num_subaps = 0;
  if (getline(is, line)) {
    iss.str(line);
    iss >> num_subaps;
  }

  for (int i = 0; i < num_subaps; i++) {
    if (getline(is, line)) {
      iss.clear();
      iss.str(line);
      double x, y, r;
      iss >> x >> y >> r;
      if (!iss.fail()) aps.emplace_back(x, y, r);
    } else break;
  }

  return is;
}

}  // namespace genetic
