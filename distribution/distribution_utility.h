// Distribution Utility
#ifndef DISTRIBUTION_DISTRIBUTION_UTILITY_H_
#define DISTRIBUTION_DISTRIBUTION_UTILITY_H_

#include <cmath>
#include "distribution.h"

namespace pbtree {
class DistributionUtility {

 public:
  inline static double gauss(double sigma, double x) {
    double exp_val = -1 * pow(x, 2) / ( 2 * pow(sigma, 2));
    double divider = sqrt(2 * M_PI) * sigma;
    return exp(exp_val) / divider;
  }

  static std::vector<double> get_gauss_kernel(uint32_t samples, double sigma);

  static std::vector<double> gauss_smoothen(
      std::vector<double> values, double sigma, uint32_t samples);

};
}  // pbtree

#endif  // DISTRIBUTION_DISTRIBUTION_UTILITY_H_
