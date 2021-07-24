// Tree loss
#ifndef DISTRIBUTION_DISTRIBUTION_H_
#define DISTRIBUTION_DISTRIBUTION_H_

#include "stdint.h"
#include <string>
#include <vector>
#include <memory>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include "boost/numeric/ublas/matrix_sparse.hpp"
#include "boost/numeric/ublas/io.hpp"
#include "boost/math/distributions/normal.hpp"
#include "boost/math/distributions/gamma.hpp"

#include "glog/logging.h"
#include "gflags/gflags.h"

#include "Tree.pb.h"
#include "utility/utility.h"

// TODO(paleylv): Maybe we need to add hessian into loss.

namespace pbtree {
class Distribution {
 public:
  Distribution() {};
  virtual ~Distribution() {};
  virtual bool calculate_loss(
      const std::vector<double>& label_data,
      const std::vector<uint64_t>& row_index_vec,
      double* loss, std::vector<double>* distribution = nullptr) = 0;

  virtual bool set_tree_node_param(
      const std::vector<double>& label_data,
      const std::vector<uint64_t>& row_index_vec,
      PBTree_Node* node) = 0;

  virtual bool plot_distribution_curve(
      const std::vector<double>& distribution,
      std::string* output_str) = 0;

  virtual bool calculate_moment(
      const PBTree_Node& node,
      double* first_moment,
      double* second_moment) = 0;

  bool calc_sample_moment(
      const std::vector<double>& label_data,
      const std::vector<uint64_t>& row_index_vec,
      double* first_moment,
      double* second_moment);

  virtual bool param_to_moment(
      const std::vector<double>& distribution,
      double* first_moment, double* second_moment) = 0;

  virtual bool calculate_boost_loss(
      const std::vector<double>& label_data,
      const std::vector<uint64_t>& record_index_vec,
      const std::vector<std::vector<double>>& prior,
      double* loss,
      const bool& evaluation = false) = 0;

  virtual bool calculate_boost_gradient(
      const std::vector<double>& label_data,
      const std::vector<uint64_t>& record_index_vec,
      const std::vector<std::vector<double>>& prior,
      std::vector<double>* likelihood) = 0;

  virtual bool set_boost_node_param(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& row_index_vec,
    const std::vector<std::vector<double>>& prior,
    PBTree_Node* node) = 0;

  virtual bool transform_param(
      const std::vector<double>& raw_dist,
      std::vector<double>* pred_dist) {
    *pred_dist = raw_dist;
    return true;
  }

//   virtual bool evaluate_rmse(
//       const std::vector<double>& label_data,
//       const std::vector<uint64_t>& record_index_vec,
//       const std::vector<std::tuple<double, double, double>>& predicted_param) = 0;

  bool evaluate_rmsle(
      const std::vector<double>& label_data,
      const std::vector<uint64_t>& record_index_vec,
      const std::vector<std::vector<double>>& predicted_dist,
      double* rmsle);

  virtual bool init_param(std::vector<double>* init_dist) = 0;

  virtual bool predict_interval(
      const std::vector<double>& distribution,
      const double& lower_interval, const double& upper_upper_interval,
      double* lower_bound, double* upper_bound) = 0;

  virtual bool get_learning_rate(
      const uint64_t& round,
      const double& initial_p1_learning_rate,
      const double& initial_p2_learning_rate,
      const double& initial_p3_learning_rate,
      double* p1_learning_rate,
      double* p2_learning_rate, double* p3_learning_rate) = 0;

  void set_target_bins(std::shared_ptr<std::vector<double>> target_bins) {
    m_target_bins_ptr_ = target_bins;
  }

  void set_target_dist(std::shared_ptr<std::vector<double>> target_dist) {
    m_target_dist_ptr_ = target_dist;
  }

 protected:
  std::shared_ptr<std::vector<double>> m_target_bins_ptr_;
  std::shared_ptr<std::vector<double>> m_target_dist_ptr_;
};

class DistributionManager {
 public:
  static std::shared_ptr<Distribution> get_distribution(PBTree_DistributionType type);
};

}  // namespace pbtree
#endif  // DISTRIBUTION_DISTRIBUTION_H_
