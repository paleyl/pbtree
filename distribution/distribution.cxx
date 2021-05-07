#include "math.h"
#include "distribution.h"
#include "gamma_distribution.h"
#include "normal_distribution.h"

DEFINE_uint32(distribution_sample_point_num, 100, "");
DEFINE_double(regularization_param, 0.1, "");
DEFINE_double(min_prob, 1e-100, "");
DEFINE_double(min_value, 1e-100, "");
DEFINE_double(max_value, 1e+100, "");
DEFINE_double(confidence_lower_bound, 0.15, "");
DEFINE_double(confidence_upper_bound, 0.85, "");
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

bool Distribution::evaluate_rmsle(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& record_index_vec,
    const std::vector<std::tuple<double, double, double>>& predicted_param,
    double* rmsle) {
  for (unsigned long i = 0; i < record_index_vec.size(); ++i)  {
    uint64_t record_index = record_index_vec[i];
    double p1 = 0, p2 = 0;
    double first_moment = 0, second_moment = 0;
    transform_param(
      std::get<0>(predicted_param[record_index]), std::get<1>(predicted_param[record_index]),
      std::get<2>(predicted_param[record_index]),
      &p1, &p2, nullptr);
    param_to_moment(std::make_tuple(p1, p2, 0), &first_moment, &second_moment);
    *rmsle += pow(log(label_data[record_index] + 1) - log(first_moment + 1), 2);
  }
  *rmsle = sqrt(*rmsle / record_index_vec.size());
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

}  // namespace pbtree
