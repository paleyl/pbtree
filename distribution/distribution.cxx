#include "distribution.h"

namespace pbtree{

double NormalDistribution::caculate_loss(
    const std::shared_ptr<boost::numeric::ublas::mapped_matrix<double>>& matrix_ptr,
    const std::vector<double>& train_data,
    const uint64_t& col_index,
    const std::vector<uint64_t>& row_index_vec) {
  return 0.0;
}
}