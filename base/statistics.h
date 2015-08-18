// File Description
// Author: Philip Salvaggio

#ifndef STATISTICS_H
#define STATISTICS_H

namespace mats {

double TConfidenceInterval(double stddev, int size, double p);
double ZConfidenceInterval(double stddev, int size, double p);

}

#endif  // STATISTICS_H
