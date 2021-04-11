#include "math.h"
#include "distribution.h"

// DEFINE_int32(input_data_line_width, 4096, "");

namespace pbtree{

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

}