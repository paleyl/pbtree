#include <algorithm>
#include "nonparametric_continuous_distribution.h"

DECLARE_double(learning_rate1);
DEFINE_string(nonparam_devivative_mode, "crps", "");
DEFINE_double(nonparam_log_freq, 1e-9, "");
// DEFINE_double(soft_evidence_ratio, 0.01, "");
// DEFINE_double(soft_evidence_gaussian_blur_ratio, 0.5, "");

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
  std::stringstream ss;
  for (unsigned int i = 1; i < m_target_bins_ptr_->size(); ++i) {
    ss << (m_target_bins_ptr_->at(i) + m_target_bins_ptr_->at(i - 1)) / 2.0
       << " " << distribution[i] / ( (m_target_bins_ptr_->at(i) - m_target_bins_ptr_->at(i - 1)) ) << "\n";
  }
  *output_str = ss.str();
  return true;
}

bool NonparametricContinousDistribution::calculate_moment(
    const PBTree_Node& node,
    double* first_moment,
    double* second_moment) {
  return true;
}

bool NonparametricContinousDistribution::update_instance(
    const PBTree_Node& node, std::vector<double>* pred_vec) {
  CHECK(unsigned(node.target_dist_size()) == pred_vec->size());
  for (unsigned int i = 0; i < pred_vec->size(); ++i) {
    pred_vec->at(i) += node.target_dist(i);
  }
  return true;
}

bool NonparametricContinousDistribution::param_to_moment(
    const std::vector<double>& distribution,
    double* first_moment, double* second_moment) {
  for (unsigned int i = 1; i < distribution.size(); ++i) {
    *first_moment += m_target_bins_ptr_->at(i - 1) * distribution[i];
    *second_moment += pow(m_target_bins_ptr_->at(i - 1) , 2) * distribution[i];
  }
  *second_moment -= pow(*first_moment, 2);
  return true;
}

bool NonparametricContinousDistribution::init_param(std::vector<double>* init_dist) {
  init_dist->resize(m_target_dist_ptr_->size());
  for (unsigned int i = 0; i < m_target_dist_ptr_->size(); ++i) {
    init_dist->at(i) = log(m_target_dist_ptr_->at(i));
  }
  return true;
}

bool NonparametricContinousDistribution::transform_param(
    const std::vector<double>& raw_dist,
    std::vector<double>* pred_dist) {
  pred_dist->resize(raw_dist.size());
  double sum = 0;
  for (unsigned int i = 0; i < raw_dist.size(); ++i) {
    pred_dist->at(i) = exp(raw_dist[i]);
    sum += pred_dist->at(i);
  }
  for (unsigned int i = 0; i < raw_dist.size(); ++i) {
    pred_dist->at(i) /= sum;
  }
  return true;
}

bool NonparametricContinousDistribution::predict_interval(
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

bool NonparametricContinousDistribution::get_learning_rate(
    const uint64_t& round,
    const double& initial_p1_learning_rate,
    const double& initial_p2_learning_rate,
    const double& initial_p3_learning_rate,
    double* p1_learning_rate,
    double* p2_learning_rate, double* p3_learning_rate) {
  // pass
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
  *loss = 0;
  std::vector<double> likelihood;
  if (!evaluation) {
    calculate_boost_gradient(
        label_data, record_index_vec, prior, &likelihood);
    std::vector<double> tmp_posterior;
    tmp_posterior.resize(m_target_dist_ptr_->size());
    for (unsigned int i = 0; i < record_index_vec.size(); ++i) {
      uint64_t record_index = record_index_vec[i];
      for (unsigned int j = 0; j < tmp_posterior.size(); ++j) {
        tmp_posterior[j] = prior[record_index][j] + likelihood[j];
      }
      std::vector<double> posterior;
      transform_param(tmp_posterior, &posterior);
      double tmp_crps;
      evaluate_one_instance_crps(label_data[record_index], posterior, &tmp_crps);
      *loss += tmp_crps;
    }
  } else {
    for (unsigned int i = 0; i < record_index_vec.size(); ++i) {
      uint64_t record_index = record_index_vec[i];
      double tmp_crps;
      std::vector<double> posterior;
      transform_param(prior[record_index], &posterior);
      evaluate_one_instance_crps(label_data[record_index], posterior, &tmp_crps);
      *loss += tmp_crps;
    }
  }
  *loss /= record_index_vec.size();
  return true;
}

bool NonparametricContinousDistribution::set_boost_node_param(
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
  node->set_distribution_type(PBTree_DistributionType_NONPARAMETRIC_CONTINUOUS);
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
// bool NonparametricContinousDistribution::calculate_boost_gradient(
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

double get_bin_width(const std::vector<double>& bins, uint32_t index) {
  if (index == 0) {
    return bins[1] - bins[0];
  } else if (index == bins.size()) {
    return bins[index - 1] - bins[index - 2];
  } else {
    return bins[index] - bins[index - 1];
  }
}

/**
 * @brief  Calculate the likelihood and used as gradient
 * @note   Use the parameter soft_evidence_ratio like learning ratio
 * @param  label_data: input, the input labels
 * @param  record_index_vec: record used to 
 * @param  prior: input, prior distribution
 * @param  likelihood: output, the likelihood distribution
 * @retval 
 */
bool NonparametricContinousDistribution::calculate_boost_gradient(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& record_index_vec,
    const std::vector<std::vector<double>>& prior,
    std::vector<double>* likelihood) {
  *likelihood = std::vector<double>(m_target_dist_ptr_->size(), 0.0);
  for (unsigned int i = 0; i < record_index_vec.size(); ++i) {
    uint32_t record_index = record_index_vec[i];
    const std::vector<double>& record_data = prior[record_index];
    std::vector<double> exp_record_data;
    exp_record_data.resize(record_data.size());
    double exp_record_sum = 0;

    for (unsigned int j = 0; j < record_data.size(); ++j) {
      exp_record_data[j] = exp(record_data[j]);
      exp_record_sum += exp_record_data[j];
    }
    std::vector<double> one_instance_gradient(m_target_dist_ptr_->size(), 0);
    // std::stringstream ss;
    // for (unsigned int j = 0; j < record_data.size(); ++j) {
    //   ss << record_data[j] << ":" << exp_record_data[j] << ":" << exp_record_data[j] / exp_record_sum << ",";
    // }
    // LOG_EVERY_N(INFO, 100000) << "Record data " << ss.str();
    uint32_t bin_index = 0;
    Utility::find_bin_index(*m_target_bins_ptr_, label_data[record_index], &bin_index);
    if (FLAGS_nonparam_devivative_mode == "mle") {
      // derivative of MLE
      for (unsigned int j = 0; j < record_data.size(); ++j) {
        if (j == bin_index) {
          one_instance_gradient[j] += -1 * (1 - exp_record_data[j] / exp_record_sum);
        } else {
          one_instance_gradient[j] += exp_record_data[j] / exp_record_sum;
        }
      }
    } else if (FLAGS_nonparam_devivative_mode == "crps_to_y") {
      // derivative of CRPS to y
      double cum = 0;
      for (unsigned int j = 0; j < record_data.size(); ++j) {
        cum += exp_record_data[j];
        if (j == 0 || m_target_bins_ptr_->at(j - 1) < label_data[record_index]) {
          one_instance_gradient[j] += 2 * (cum / exp_record_sum) * exp_record_data[j] / exp_record_sum;
        } else {
          one_instance_gradient[j] += 2 * (cum / exp_record_sum - 1) * exp_record_data[j] / exp_record_sum;
        }
      }
    } else if (FLAGS_nonparam_devivative_mode == "crps") {
      // direct derivative of CRPS
      double cum = 0;
      double cube_exp_sum = pow(exp_record_sum, 3);
      for (unsigned int j = 0; j < record_data.size(); ++j) {  // M steps for cdf
        cum += exp_record_data[j];
        for (unsigned int k = 0; k < record_data.size(); ++k) {  // crps derivate of M values
          double tmp_grad = 0;
          if (j == 0 || m_target_bins_ptr_->at(j - 1) < label_data[record_index]) {
            if (k <= j) {
              tmp_grad = 2 * cum * (exp_record_sum - cum) / cube_exp_sum;  // 2(X+A)(B-A) / (X+B)^3
            } else {
              tmp_grad = -2 * pow(cum, 2) / cube_exp_sum;  // -2 * C^2 / (X+B)^3
            }
          } else {
            if (k <= j) {
              tmp_grad = -2 * pow(exp_record_sum - cum, 2) / cube_exp_sum;  // -2(A-B)^2 / (X+B)^3 
            } else {
              tmp_grad = 2 * cum * (exp_record_sum - cum) / cube_exp_sum;  // 2 * C * (X+B-C) / (X+B)^3
            }
          }
          // tmp_grad *= exp_record_data[k];
          tmp_grad *= (exp_record_data[k] * get_bin_width(*m_target_bins_ptr_, k));
          one_instance_gradient[k] += tmp_grad;
          if (std::isnan(tmp_grad) || std::isinf(tmp_grad)) {
            LOG_EVERY_N(INFO, 1000000) << "Nan grad " << label_data[record_index] << " j = " << j << ", k = " << k;
          }
          // if (k == record_data.size() - 1) {
          //   LOG_EVERY_N(INFO, 10000) << "Tmp grad = " << tmp_grad << ", width = " << get_bin_width(*m_target_bins_ptr_, k);
          // }
        }
      }
    } else {
      LOG(FATAL) << "Unkonw devivative_mode " << FLAGS_nonparam_devivative_mode;
    }
    for (unsigned int j = 0; j < likelihood->size(); ++j) {
      likelihood->at(j) += one_instance_gradient[j];
    }

    // DEBUG LOGS
    if (rand() * 1.0 / RAND_MAX < FLAGS_nonparam_log_freq
        || std::isnan(one_instance_gradient.back())) {
      std::stringstream ss1, ss2;
      for (unsigned int j = 0; j <= bin_index; ++j) {
        ss1 << one_instance_gradient[j] << ",";
      }
      for (unsigned int j = bin_index + 1; j < one_instance_gradient.size(); ++j) {
        ss2 << one_instance_gradient[j] << ",";
      }
      LOG(INFO) << "Label = " << label_data[record_index]
                << ", bin_index = " << bin_index
                << ", left gradient " << ss1.str()
                << ", right gradient " << ss2.str();
    }
  }
  for (unsigned int i = 0; i < likelihood->size(); ++i) {
    likelihood->at(i) *= (-1 * FLAGS_learning_rate1 / record_index_vec.size());
  }
  if (rand() * 1.0 / RAND_MAX < 0.001) {
    std::stringstream ss;
    for (unsigned int i = 0; i < likelihood->size(); ++i) {
      ss << likelihood->at(i) << ",";
    }
    LOG(INFO) << "Gradient " << ss.str();
  }

  return true;
}

}  // pbtree
