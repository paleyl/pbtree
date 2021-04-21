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
#include "boost/math/distributions/normal.hpp"

#include "glog/logging.h"
#include "gflags/gflags.h"

#include "Tree.pb.h"
#include "utility/utility.h"

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

  virtual bool plot_distribution_curve(
      const double& p1,
      const double& p2,
      const double& p3,
      std::string* output_str) = 0;
};

class DistributionManager {
 public:
  static std::shared_ptr<Distribution> get_distribution(PBTree_DistributionType type);
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

  bool plot_distribution_curve(
      const double& p1,
      const double& p2,
      const double& p3,
      std::string* output_str);

  void print_version() {
    VLOG(202) << "Normal distribution";
  }
};

class GammaDistribution : public Distribution {
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

  bool plot_distribution_curve(
      const double& p1,
      const double& p2,
      const double& p3,
      std::string* output_str);

  void print_version() {
    VLOG(202) << "Gamma distribution";
  }
};

}  // namespace pbtree
#endif  // LOSS_LOSS_H_