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

#include "glog/logging.h"

namespace pbtree {
class Distribution {
 public:
  Distribution() {};
  virtual ~Distribution() {};
  virtual double calculate_loss(
      const std::vector<double>& label_data,
      const uint64_t& col_index,
      const std::vector<uint64_t>& row_index_vec,
      const double& split_point) {
        return 0;
  };
  // virtual double mean() = 0;
};

class NormalDistribution : public Distribution {
 public:
  double calculate_loss(
      const std::vector<double>& train_data,
      const uint64_t& col_index,
      const std::vector<uint64_t>& row_index_vec);
  void print_version() {
    VLOG(202) << "Normal distribution";
  }
};
}  // namespace pbtree
#endif  // LOSS_LOSS_H_