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
      std::tuple<double, double, double>& param,
      double* first_moment, double* second_moment) = 0;

  virtual bool calculate_boost_loss(
      const std::vector<double>& label_data,
      const std::vector<uint64_t>& record_index_vec,
      const std::vector<std::tuple<double, double, double>>& predicted_param,
      double* loss,
      const bool& evaluation = false) = 0;

  virtual bool calculate_boost_gradient(
      const std::vector<double>& label_data,
      const std::vector<uint64_t>& record_index_vec,
      const std::vector<std::tuple<double, double, double>>& predicted_param,
      double* g_p1, double* g_p2, double* g_p3) = 0;

  virtual bool set_boost_node_param(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& row_index_vec,
    const std::vector<std::tuple<double, double, double>>& predicted_param,
    PBTree_Node* node) = 0;

  virtual bool transform_param(
      const double& raw_p1, const double& raw_p2, const double& raw_p3,
      double* p1, double* p2, double* p3) {
    *p1 = raw_p1;
    *p2 = raw_p2;
    if (p3) *p3 = raw_p3;
    return true;
  }

//   virtual bool evaluate_rmse(
//       const std::vector<double>& label_data,
//       const std::vector<uint64_t>& record_index_vec,
//       const std::vector<std::tuple<double, double, double>>& predicted_param) = 0;

//   virtual bool evaluate_rmsle(
//       const std::vector<double>& label_data,
//       const std::vector<uint64_t>& record_index_vec,
//       const std::vector<std::tuple<double, double, double>>& predicted_param) = 0;

  virtual bool init_param(double* p1, double* p2, double* p3) = 0;
};

class DistributionManager {
 public:
  static std::shared_ptr<Distribution> get_distribution(PBTree_DistributionType type);
};

}  // namespace pbtree
#endif  // DISTRIBUTION_DISTRIBUTION_H_
