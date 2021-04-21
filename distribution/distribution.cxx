#include "math.h"
#include "distribution.h"

DEFINE_uint32(distribution_sample_point_num, 100, "");
// DEFINE_int32(input_data_line_width, 4096, "");

namespace pbtree {

bool NormalDistribution::calculate_loss(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& row_index_vec,
    double* loss, double* p1 /*= nullptr*/, double* p2 /*= nullptr*/, double* p3 /*= nullptr*/) {
  double mu = 0;
  double sigma = 0;
  double square_sum = 0;
  // Compute mu and sigma
  for (auto iter = row_index_vec.begin(); iter < row_index_vec.end(); ++iter) {
    mu += label_data[*iter];
    square_sum += label_data[*iter] * label_data[*iter];
  }
  mu /= row_index_vec.size();
  // E(X^2) - (EX)^2
  double variance = square_sum / row_index_vec.size() - mu * mu;
  if (!Utility::check_double_le(0, variance)) {
    sigma = -1;
  } else if (Utility::check_double_equal(0, variance)) {
    sigma = 0;
  } else {
    sigma = sqrt(variance);
  }
  // VLOG(102) << sigma;
  // Compute loss
  double tmp_loss = 0;
  for (auto iter = row_index_vec.begin(); iter < row_index_vec.end(); ++iter) {
    tmp_loss += (mu - label_data[*iter]) * (mu - label_data[*iter]);
  }
  tmp_loss /= row_index_vec.size();
  *loss = tmp_loss;
  if (p1 != nullptr) *p1 = mu;
  if (p2 != nullptr) *p2 = sigma;
  return true;
}

bool NormalDistribution::set_tree_node_param(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& row_index_vec,
    PBTree_Node* node) {
  double mu = 0;
  double sigma = 0;
  double square_sum = 0;
  // Compute mu and sigma
  for (auto iter = row_index_vec.begin(); iter < row_index_vec.end(); ++iter) {
    mu += label_data[*iter];
    square_sum += label_data[*iter] * label_data[*iter];
  }
  mu /= row_index_vec.size();
  // E(X^2) - (EX)^2
  double variance = square_sum / row_index_vec.size() - mu * mu;
  if (!Utility::check_double_le(0, variance)) {
    sigma = -1;
  } else if (Utility::check_double_equal(0, variance)) {
    sigma = 0;
  } else {
    sigma = sqrt(variance);
  }
  sigma = sqrt(variance);
  node->set_p1(mu);
  node->set_p2(sigma);
  node->set_distribution_type(PBTree_DistributionType_NORMAL_DISTRIBUITION);
  return true;
}

bool NormalDistribution::plot_distribution_curve(
    const double& p1, const double& p2,
    const double& p3,
    std::string* output_str) {
  boost::math::normal_distribution<double> dist(p1, p2);
  
  const double lower_bound = p1 - 5 * p2;
  const double upper_bound = p1 + 5 * p2;
  const double step = (upper_bound - lower_bound) / FLAGS_distribution_sample_point_num;
  std::stringstream ss;
  for (unsigned int i = 0; i < FLAGS_distribution_sample_point_num; ++i) {
    double x = i * step + lower_bound;
    double y = boost::math::pdf(dist, x);
    ss << x << " " << y << "\n";
  }
  *output_str = ss.str();
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

bool GammaDistribution::calculate_loss(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& row_index_vec,
    double* loss, double* p1 /*= nullptr*/, double* p2 /*= nullptr*/, double* p3 /*= nullptr*/) {
  return true;
}

bool GammaDistribution::set_tree_node_param(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& row_index_vec,
    PBTree_Node* node) {
  return true;
}

bool GammaDistribution::plot_distribution_curve(
    const double& p1, const double& p2,
    const double& p3,
    std::string* output_str) {
  return true;
}

}  // pbtree
