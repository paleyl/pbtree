// utility
#ifndef UTILITY_UTILITY_H_
#define UTILITY_UTILITY_H_

#include "stdint.h"
#include <algorithm>
#include <string>
#include <vector>

namespace pbtree {

// typedef std::pair<uint64_t, double> feature_t;
// typedef std::vector<feature_t> featurevec_t;
// typedef std::pair<double, featurevec_t> instance_t;
// typedef std::vector<instance_t> label_data_t;
// typedef std::vector<featurevec_t> predict_data_t;

class Utility {
 public:
  static bool split_string(
      const std::string &src, const char *separators,
      std::vector<std::string> *result);
  static bool check_double_equal(const double& a, const double& b);
  static bool check_double_le(const double& a, const double& b);  // less or equal
  static bool find_bin_index(
      const std::vector<double>& bins, const double& target, uint32_t* index);
};
}  // namespace pbtree

#endif  // UTILITY_UTILITY_H_
