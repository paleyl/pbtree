#include "math.h"
#include "distribution.h"
#include "gamma_distribution.h"
#include "normal_distribution.h"

DEFINE_uint32(distribution_sample_point_num, 100, "");
DEFINE_double(regularization_param, 0.1, "");
DEFINE_double(min_prob, 1e-100, "");
// DEFINE_int32(input_data_line_width, 4096, "");

namespace pbtree {

bool Distribution::calc_sample_moment(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& row_index_vec,
    double* first_moment,
    double* second_moment) {
  double mu = 0;
  double square_sum = 0;
  // Compute mu and sigma
  for (auto iter = row_index_vec.begin(); iter < row_index_vec.end(); ++iter) {
    mu += label_data[*iter];
    square_sum += label_data[*iter] * label_data[*iter];
  }
  mu /= row_index_vec.size();
  *first_moment = mu;
  // E(X^2) - (EX)^2
  *second_moment = square_sum / row_index_vec.size() - mu * mu;
  return true;
}

std::shared_ptr<Distribution> DistributionManager::get_distribution(PBTree_DistributionType type) {
  std::shared_ptr<Distribution> distribution_ptr;
  switch (type)
  {
  case PBTree_DistributionType_NORMAL_DISTRIBUITION:
    distribution_ptr = std::shared_ptr<Distribution>(new NormalDistribution());
    break;
  case PBTree_DistributionType_GAMMA_DISTRIBUTION:
    distribution_ptr = std::shared_ptr<Distribution>(new GammaDistribution());
    break;
  default:
    LOG(FATAL) << "Unrecognized distribution type " << type;
    break;
  }
  return distribution_ptr;
}

}  // pbtree
