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
#include "gflags/gflags.h"

#include "Tree.pb.h"

namespace pbtree {
class Distribution {
 public:
  Distribution() {};
  virtual ~Distribution() {};
  virtual bool calculate_loss(
      const std::vector<double>& label_data,
      const std::vector<uint64_t>& row_index_vec,
      double* loss, double* p1 = nullptr,
      double* p2 = nullptr, double* p3 = nullptr) = 0;

  virtual bool set_tree_node_param(
      const std::vector<double>& label_data,
      const std::vector<uint64_t>& row_index_vec,
      PBTree_Node* node) = 0;
};

class NormalDistribution : public Distribution {
 public:
  bool calculate_loss(
      const std::vector<double>& train_data,
      const std::vector<uint64_t>& row_index_vec,
      double* loss, double* p1 = nullptr,
      double* p2 = nullptr, double* p3 = nullptr);

  bool set_tree_node_param(
      const std::vector<double>& label_data,
      const std::vector<uint64_t>& row_index_vec,
      PBTree_Node* node);

  void print_version() {
    VLOG(202) << "Normal distribution";
  }
};
}  // namespace pbtree
#endif  // LOSS_LOSS_H_