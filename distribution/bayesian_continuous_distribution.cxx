#include <algorithm>
#include "bayesian_continuous_distribution.h"

DECLARE_double(learning_rate1);
DEFINE_double(soft_evidence_ratio, 0.01, "");
DEFINE_double(soft_evidence_gaussian_blur_ratio, 0.5, "");

namespace pbtree {

bool BayesianContinuousDistribution::set_tree_node_param(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& row_index_vec,
    PBTree_Node* node) {
  return true;
}

bool BayesianContinuousDistribution::plot_distribution_curve(
    const std::vector<double>& distribution,
    std::string* output_str) {
  std::stringstream ss;
  for (unsigned int i = 1; i < m_target_bins_ptr_->size(); ++i) {
    ss << (m_target_bins_ptr_->at(i) + m_target_bins_ptr_->at(i - 1)) / 2.0
       << " " << distribution[i] / ( (m_target_bins_ptr_->at(i) - m_target_bins_ptr_->at(i - 1)) ) << "\n";
  }
  *output_str = ss.str();
  return true;
}

bool BayesianContinuousDistribution::calculate_moment(
    const PBTree_Node& node,
    double* first_moment,
    double* second_moment) {
  return true;
}

// bool find_bin_index(
//     const std::vector<double>& bins, const double& target, uint32_t* index) {
//   auto pos = std::upper_bound(bins.data(), bins.data() + bins.size(), target);
//   *index = pos - bins.data();
//   return true;
// }

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

bool BayesianContinuousDistribution::update_instance(
    const PBTree_Node& node, std::vector<double>* pred_vec) {
  CHECK(unsigned(node.target_dist_size()) == pred_vec->size());
  std::vector<double> likelihood(node.target_dist().begin(), node.target_dist().end());
  std::vector<double> posterior;
  calculate_posterior(*pred_vec, likelihood, &posterior);
  *pred_vec = posterior;
  return true;
}

bool BayesianContinuousDistribution::param_to_moment(
    const std::vector<double>& distribution,
    double* first_moment, double* second_moment) {
  for (unsigned int i = 1; i < distribution.size(); ++i) {
    *first_moment += m_target_bins_ptr_->at(i - 1) * distribution[i];
    *second_moment += pow(m_target_bins_ptr_->at(i - 1) , 2) * distribution[i];
  }
  *second_moment -= pow(*first_moment, 2);
  return true;
}

bool BayesianContinuousDistribution::init_param(std::vector<double>* init_dist) {
  *init_dist = *m_target_dist_ptr_;
  return true;
}

bool BayesianContinuousDistribution::transform_param(
    const std::vector<double>& raw_dist,
    std::vector<double>* pred_dist) {
  *pred_dist = raw_dist;
  return true;
}

bool BayesianContinuousDistribution::predict_interval(
    const std::vector<double>& distribution,
    const double& lower_interval, const double& upper_interval,
    double* lower_bound, double* upper_bound) {
  uint32_t lower_index, upper_index;
  std::vector<double> cdf;
  pdf_to_cdf(distribution, &cdf);
  Utility::find_bin_index(cdf, lower_interval, &lower_index);
  Utility::find_bin_index(cdf, upper_interval, &upper_index);
  if (lower_index == 0) lower_index = 1;
  *lower_bound = m_target_bins_ptr_->at(lower_index - 1);
  *upper_bound = m_target_bins_ptr_->at(upper_index - 1);
  return true;
}

bool BayesianContinuousDistribution::get_learning_rate(
    const uint64_t& round,
    const double& initial_p1_learning_rate,
    const double& initial_p2_learning_rate,
    const double& initial_p3_learning_rate,
    double* p1_learning_rate,
    double* p2_learning_rate, double* p3_learning_rate) {
  // pass
  return true;
}

bool BayesianContinuousDistribution::calculate_loss(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& row_index_vec,
    double* loss, std::vector<double>* distribution) {
  return true;
}

    // std::vector<double> tmp_posterior;
    // tmp_posterior.resize(likelihood.size());
    // for (unsigned int i = 0; i < record_index_vec.size(); ++i) {
    //   for (unsigned int j = 0; j < likelihood.size(); ++j) {
    //     tmp_posterior[j] = prior[i][j] + likelihood[j];
    //   }
    //   posterior[i] = tmp_posterior;
    // }

bool BayesianContinuousDistribution::calculate_boost_loss(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& record_index_vec,
    const std::vector<std::vector<double>>& prior,
    double* loss,
    const bool& evaluation) {
  *loss = 0;
  std::vector<double> likelihood;
  if (!evaluation) {
    calculate_boost_gradient(
        label_data, record_index_vec, prior, &likelihood);
    std::vector<double> tmp_posterior;
    for (unsigned int i = 0; i < record_index_vec.size(); ++i) {
      uint64_t record_index = record_index_vec[i];
      calculate_posterior(prior[record_index], likelihood, &tmp_posterior);
      double tmp_crps;
      evaluate_one_instance_crps(label_data[record_index], tmp_posterior, &tmp_crps);
      *loss += tmp_crps;
    }
  } else {
    evaluate_crps(label_data, record_index_vec, prior, loss);
  }
  *loss /= record_index_vec.size();
  return true;
}

bool BayesianContinuousDistribution::set_boost_node_param(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& row_index_vec,
    const std::vector<std::vector<double>>& prior,
    PBTree_Node* node) {
  std::vector<double> likelihood;
  calculate_boost_gradient(
      label_data, row_index_vec, prior, &likelihood);
  node->clear_target_dist();
  for (unsigned int i = 0; i < likelihood.size(); ++i)
    node->add_target_dist(likelihood[i]);
  node->set_distribution_type(PBTree_DistributionType_BAYESIAN_CONTINUOUS);
  return true;
}

// /**
//  * @brief  Calculate the gradient w.r.t. CPRS
//  * @note   
//  * @param  label_data: 
//  * @param  record_index_vec: 
//  * @param  prior: 
//  * @param  likelihood: 
//  * @retval 
//  */
// bool BayesianContinuousDistribution::calculate_boost_gradient(
//     const std::vector<double>& label_data,
//     const std::vector<uint64_t>& record_index_vec,
//     const std::vector<std::vector<double>>& prior,
//     std::vector<double>* likelihood) {
//   // likelihood->resize(m_target_dist_ptr_->size());
//   *likelihood = std::vector<double>(m_target_dist_ptr_->size(), 0.0);
//   for (uint32_t i = 0; i < record_index_vec.size(); ++i) {
//     uint64_t record_index = record_index_vec[i];
//     std::vector<double> tmp_cdf;
//     pdf_to_cdf(prior[record_index], &tmp_cdf);
//     double tmp_gradient = 0;
//     for (uint32_t j = tmp_cdf.size() - 1; j > 0; --j) {
//       double real_cdf = label_data[record_index] > m_target_bins_ptr_->at(j - 1) ? 1.0 : 0.0;
//       tmp_gradient += 2 * ( tmp_cdf[j] - real_cdf);
//       likelihood->at(j) += tmp_gradient;
//     }
//     likelihood->at(0) += ( tmp_gradient + 2 * tmp_cdf[0]);
//   }
//   for (uint32_t i = 0; i < likelihood->size(); ++i) {
//     likelihood->at(i) *= ( FLAGS_learning_rate1 / record_index_vec.size());
//   }
//   return true;
// }

/**
 * @brief  Calculate the likelihood and used as gradient
 * @note   Use the parameter soft_evidence_ratio like learning ratio
 * @param  label_data: input, the input labels
 * @param  record_index_vec: record used to 
 * @param  prior: input, prior distribution
 * @param  likelihood: output, the likelihood distribution
 * @retval 
 */
bool BayesianContinuousDistribution::calculate_boost_gradient(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& record_index_vec,
    const std::vector<std::vector<double>>& prior,
    std::vector<double>* likelihood) {
  // likelihood->resize(m_target_dist_ptr_->size());
  std::vector<double> tmp_likelihood = std::vector<double>(m_target_dist_ptr_->size(), 0.0);
  for (uint32_t i = 0; i < record_index_vec.size(); ++i) {
    uint32_t index = 0;
    Utility::find_bin_index(*m_target_bins_ptr_, label_data[record_index_vec[i]], &index);
    tmp_likelihood[index] += 1.0 / record_index_vec.size();
  }
  // for (uint32_t i = 0; i < tmp_likelihood.size(); ++i) {
  //   tmp_likelihood[i] = tmp_likelihood[i] / m_target_dist_ptr_->at(i);
  // }
  // Use sort of soft evidence
  // for (uint32_t i = 0; i < likelihood->size(); ++i) {
  //   (*likelihood)[i] *= (1 - FLAGS_soft_evidence_ratio);
  //   (*likelihood)[i] += FLAGS_soft_evidence_ratio / likelihood->size();
  // }
  uint32_t samples = m_target_dist_ptr_->size() * FLAGS_soft_evidence_gaussian_blur_ratio;
  *likelihood = DistributionUtility::gauss_smoothen(tmp_likelihood, 1.0, samples);
  return true;
}

}  // pbtree
