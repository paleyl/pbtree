#include <algorithm>
#include "nonparametric_continuous_distribution.h"

namespace pbtree {

bool NonparametricContinousDistribution::set_tree_node_param(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& row_index_vec,
    PBTree_Node* node) {
  return true;
}

bool NonparametricContinousDistribution::plot_distribution_curve(
    const std::vector<double>& distribution,
    std::string* output_str) {
  return true;
}

bool NonparametricContinousDistribution::calculate_moment(
    const PBTree_Node& node,
    double* first_moment,
    double* second_moment) {
  return true;
}

bool NonparametricContinousDistribution::param_to_moment(
    const std::vector<double>& distribution,
    double* first_moment, double* second_moment) {
  return true;
}

bool NonparametricContinousDistribution::init_param(std::vector<double>* init_dist) {
  *init_dist = *m_target_dist_ptr_;
  return true;
}

bool NonparametricContinousDistribution::transform_param(
    const std::vector<double>& raw_dist,
    std::vector<double>* pred_dist) {
  return true;
}

bool NonparametricContinousDistribution::predict_interval(
    const std::vector<double>& distribution,
    const double& lower_interval, const double& upper_interval,
    double* lower_bound, double* upper_bound) {
  return true;
}

bool NonparametricContinousDistribution::get_learning_rate(
    const uint64_t& round,
    const double& initial_p1_learning_rate,
    const double& initial_p2_learning_rate,
    const double& initial_p3_learning_rate,
    double* p1_learning_rate,
    double* p2_learning_rate, double* p3_learning_rate) {
  return true;
}

bool NonparametricContinousDistribution::calculate_loss(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& row_index_vec,
    double* loss, std::vector<double>* distribution) {
  return true;
}

bool NonparametricContinousDistribution::calculate_boost_loss(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& record_index_vec,
    const std::vector<std::vector<double>>& prior,
    double* loss,
    const bool& evaluation) {

  return true;
}

bool NonparametricContinousDistribution::set_boost_node_param(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& row_index_vec,
    const std::vector<std::vector<double>>& likelihood,
    PBTree_Node* node) {
  return true;
}

bool find_bin_index(
    const std::vector<double>& bins, const double& target, uint32_t* index) {
  auto pos = std::upper_bound(bins.data(), bins.data() + bins.size(), target);
  *index = pos - bins.data();
  return true;
}

bool calculate_posterior(
    const std::vector<double>& prior, const std::vector<double>& likelihood,
    std::vector<double>* posterior) {
  CHECK_EQ(prior.size(), likelihood.size());
  posterior->resize(prior.size());
  double sum_product = 0;
  for (uint32_t i = 0; i < prior.size(); ++i) {
    (*posterior)[i] = prior[i] * likelihood[i];
    sum_product += (*posterior)[i];
  }
  for (uint32_t i = 0; i < prior.size(); ++i) {
    (*posterior)[i] /= sum_product;
  }
  return true;
}

bool NonparametricContinousDistribution::calculate_boost_gradient(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& record_index_vec,
    const std::vector<std::vector<double>>& prior,
    std::vector<double>* likelihood) {
  // likelihood->resize(m_target_dist_ptr_->size());
  *likelihood = std::vector<double>(m_target_dist_ptr_->size(), 0.0);
  for (uint32_t i = 0; i < record_index_vec.size(); ++i) {
    uint32_t index = 0;
    find_bin_index(*m_target_bins_ptr_, record_index_vec[i], &index);
    (*likelihood)[index] += 1.0 / record_index_vec.size();
  }
  return true;
}

}  // pbtree
