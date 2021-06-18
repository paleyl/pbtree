#ifndef DISTRIBUTION_NORMAL_DISTRIBUTION_H_
#define DISTRIBUTION_NORMAL_DISTRIBUTION_H_

#include "distribution/distribution.h"

namespace pbtree {

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

  bool calculate_moment(
      const PBTree_Node& node,
      double* first_moment,
      double* second_moment);

  bool param_to_moment(
      const std::tuple<double, double, double>& param,
      double* first_moment, double* second_moment);

  bool init_param(double* p1, double* p2, double* p3) {
      *p1 = 0.0;  // mu
      *p2 = 1.0;  // sigma
      return true;
  }

  bool calculate_boost_loss(
      const std::vector<double>& label_data,
      const std::vector<uint64_t>& record_index_vec,
      const std::vector<std::tuple<double, double, double>>& predicted_param,
      double* loss,
      const bool& evaluation = false);

  virtual bool calculate_boost_gradient(
      const std::vector<double>& label_data,
      const std::vector<uint64_t>& record_index_vec,
      const std::vector<std::tuple<double, double, double>>& predicted_param,
      double* g_p1, double* g_p2, double* g_p3);

  bool set_boost_node_param(
      const std::vector<double>& label_data,
      const std::vector<uint64_t>& row_index_vec,
      const std::vector<std::tuple<double, double, double>>& predicted_param,
      PBTree_Node* node);

  bool evaluate_rmse(
      const std::vector<double>& label_data,
      const std::vector<uint64_t>& record_index_vec,
      const std::vector<std::tuple<double, double, double>>& predicted_param) {
    return true;
  }

  bool transform_param(
      const double& raw_p1, const double& raw_p2, const double& raw_p3,
      double* p1, double* p2, double* p3);

  bool predict_interval(
      const double& p1, const double& p2, const double& p3,
      const double& lower_interval, const double& upper_upper_interval,
      double* lower_bound, double* upper_bound);

  // bool evaluate_rmsle(
  //     const std::vector<double>& label_data,
  //     const std::vector<uint64_t>& record_index_vec,
  //     const std::vector<std::tuple<double, double, double>>& predicted_param,
  //     double* rmsle);
  bool get_learning_rate(
      const uint64_t& round,
      const double& initial_p1_learning_rate,
      const double& initial_p2_learning_rate,
      const double& initial_p3_learning_rate,
      double* p1_learning_rate,
      double* p2_learning_rate, double* p3_learning_rate);

  void print_version() {
    VLOG(202) << "Normal distribution";
  }
};

}  // namespace pbtree

#endif  // DISTRIBUTION_NORMAL_DISTRIBUTION_H_
