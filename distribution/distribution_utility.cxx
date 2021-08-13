#include "distribution_utility.h"

namespace pbtree {

std::vector<double> DistributionUtility::get_gauss_kernel(uint32_t samples, double sigma) {
  std::vector<double> v;
  v.resize(samples);
  uint32_t steps = samples / 2;
  double step_size = 3 * sigma / steps;
  for (uint32_t i = 0; i < samples; ++i) {
    double tmp_y = gauss(sigma, i * step_size - 3 * sigma);
    v[i] = tmp_y * 6 / samples;
  }
  return v;
}

std::vector<double> DistributionUtility::gauss_smoothen(
    std::vector<double> values, double sigma, uint32_t samples) {
  std::vector<double> v;
  v.resize(values.size());
  std::vector<double> kernel = get_gauss_kernel(samples, sigma);
  for (uint32_t i = 0; i < values.size(); ++i) {
    double tmp_v = 0;
    for (uint32_t j = 0; j < samples; ++j) {
      uint32_t pos = i - samples / 2 + j;
      if (pos >= 0 && pos < values.size()) {
        tmp_v += values[pos] * kernel[j];
      }
    }
    v[i] = tmp_v;
  }
  return v;
}

}  // pbtree
