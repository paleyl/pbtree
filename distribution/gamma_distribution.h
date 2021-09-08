#ifndef DISTRIBUTION_GAMMA_DISTRIBUTION_H_
#define DISTRIBUTION_GAMMA_DISTRIBUTION_H_

#include "distribution.h"

namespace pbtree {
class GammaDistribution : public Distribution {
 public:
  bool calculate_loss(
      const std::vector<double>& label_data,
      const std::vector<uint64_t>& row_index_vec,
      double* loss, std::vector<double>* distribution = nullptr);

  bool set_tree_node_param(
      const std::vector<double>& label_data,
      const std::vector<uint64_t>& row_index_vec,
      PBTree_Node* node);

  bool plot_distribution_curve(
      const std::vector<double>& distribution,
      std::string* output_str);

  bool calculate_moment(
      const PBTree_Node& node,
      double* first_moment,
      double* second_moment);

  bool param_to_moment(
      const std::vector<double>& distribution,
      double* first_moment, double* second_moment);

  bool init_param(std::vector<double>* init_dist);

  bool calculate_boost_loss(
      const std::vector<double>& label_data,
      const std::vector<uint64_t>& record_index_vec,
      const std::vector<std::vector<double>>& prior,
      double* loss,
      const bool& evaluation = false);

  virtual bool calculate_boost_gradient(
      const std::vector<double>& label_data,
      const std::vector<uint64_t>& record_index_vec,
      const std::vector<std::vector<double>>& prior,
      std::vector<double>* likelihood);

  bool update_instance(const PBTree_Node& node, std::vector<double>* pred_vec) {
    pred_vec->at(0) += node.p1();
    pred_vec->at(1) += node.p2();
    return true;
  }

  bool set_boost_node_param(
      const std::vector<double>& label_data,
      const std::vector<uint64_t>& row_index_vec,
      const std::vector<std::vector<double>>& likelihood,
      PBTree_Node* node);

  bool transform_param(
      const std::vector<double>& raw_dist,
      std::vector<double>* pred_dist);

  bool predict_interval(
      const std::vector<double>& distribution,
      const double& lower_interval, const double& upper_interval,
      double* lower_bound, double* upper_bound);

  bool get_learning_rate(
      const uint64_t& round,
      const double& initial_p1_learning_rate,
      const double& initial_p2_learning_rate,
      const double& initial_p3_learning_rate,
      double* p1_learning_rate,
      double* p2_learning_rate, double* p3_learning_rate);

  bool evaluate_crps(
      const std::vector<double>& label_data,
      const std::vector<uint64_t>& record_index_vec,
      const std::vector<std::vector<double>>& predicted_dist,
      double* crps);

  bool evaluate_one_instance_crps(
      const double& label_data,
      const std::vector<double>& predicted_dist,
      double* crps);

  void print_version() {
    VLOG(202) << "Gamma distribution";
  }
};

}  // namespace pbtree

#endif  // DISTRIBUTION_GAMMA_DISTRIBUTION_H_
