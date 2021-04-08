#include "distribution.h"
#include "math.h"

namespace pbtree{

double NormalDistribution::caculate_loss(
    const std::vector<double>& label_data,
    const uint64_t& col_index,
    const std::vector<uint64_t>& row_index_vec) {
  double mu = 0;
  double sigma = 0;
  double square_sum = 0;
  // Compute mu and sigma
  for (auto iter = row_index_vec.begin(); iter < row_index_vec.end(); ++iter) {
    mu += label_data[*iter];
    square_sum += label_data[*iter] * label_data[*iter];
  }
  mu /= row_index_vec.size();
  // (EX)^2 - E(X^2)
  double variance = mu * mu - square_sum / row_index_vec.size();
  sigma = sqrt(variance);
  VLOG(102) << sigma;
  // Compute loss
  double loss = 0;
  for (auto iter = row_index_vec.begin(); iter < row_index_vec.end(); ++iter) {
    loss += (mu - *iter) * (mu - *iter);
  }
  loss /= row_index_vec.size();
  return loss;
}
}