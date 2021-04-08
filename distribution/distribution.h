// Tree loss
#ifndef LOSS_LOSS_H_
#define LOSS_LOSS_H_

#include "stdint.h"
#include <string>
#include <vector>
#include <memory>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include "boost/numeric/ublas/matrix_sparse.hpp"
#include "boost/numeric/ublas/io.hpp"

namespace pbtree {
class Distribution {
  virtual double caculate_loss(
      const std::shared_ptr<boost::numeric::ublas::mapped_matrix<double>>& matrix_ptr,
      const std::vector<double>& train_data,
      const uint64_t& col_index,
      const std::vector<uint64_t>& row_index_vec,
      const double& split_point) = 0;
  virtual double mean() = 0;
};

class NormalDistribution : public Distribution {
  double caculate_loss(
      const std::shared_ptr<boost::numeric::ublas::mapped_matrix<double>>& matrix_ptr,
      const std::vector<double>& train_data,
      const uint64_t& col_index,
      const std::vector<uint64_t>& row_index_vec);
};
}  // namespace pbtree
#endif  // LOSS_LOSS_H_