#include "math.h"
#include "distribution.h"
#include "gamma_distribution.h"
#include "normal_distribution.h"
#include "nonparametric_continuous_distribution.h"
#include "bayesian_continuous_distribution.h"

DEFINE_uint32(distribution_sample_point_num, 100, "");
DEFINE_double(regularization_param1, 0.01, "");
DEFINE_double(regularization_param2, 0.01, "");
DEFINE_double(learning_rate1, 0.1, "");
DEFINE_double(learning_rate2, 0.1, "");
DEFINE_double(min_prob, 1e-100, "");
DEFINE_double(min_value, 1e-100, "");
DEFINE_double(max_value, 1e+100, "");
DEFINE_double(confidence_lower_bound, 0.15, "");
DEFINE_double(confidence_upper_bound, 0.85, "");
// DEFINE_int32(input_data_line_width, 4096, "");

namespace pbtree {

bool Distribution::calc_sample_moment(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& row_index_vec,
    double* first_moment,
    double* second_moment) {
  double mu = 0;
  double square_sum = 0;
  // Compute mu and sigma
  for (auto iter = row_index_vec.begin(); iter < row_index_vec.end(); ++iter) {
    mu += label_data[*iter];
    square_sum += label_data[*iter] * label_data[*iter];
  }
  mu /= row_index_vec.size();
  *first_moment = mu;
  // E(X^2) - (EX)^2
  *second_moment = square_sum / row_index_vec.size() - mu * mu;
  return true;
}

bool Distribution::evaluate_rmsle(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& record_index_vec,
    const std::vector<std::vector<double>>& predicted_dist,
    double* rmsle) {
  for (unsigned long i = 0; i < record_index_vec.size(); ++i)  {
    uint64_t record_index = record_index_vec[i];
//    double p1 = 0, p2 = 0;
    double first_moment = 0, second_moment = 0;
    std::vector<double> transformed_dist;
    transform_param(predicted_dist[record_index],
      &transformed_dist);
    param_to_moment(transformed_dist, &first_moment, &second_moment);
    *rmsle += pow(log(label_data[record_index] + 1) - log(first_moment + 1), 2);
  }
  *rmsle = sqrt(*rmsle / record_index_vec.size());
  return true;
}

bool Distribution::pdf_to_cdf(
    const std::vector<double>& predicted_pdf,
    std::vector<double>* predicted_cdf) {
  predicted_cdf->resize(predicted_pdf.size());
  (*predicted_cdf)[0] = predicted_pdf[0];
  for (unsigned int i = 1; i < predicted_pdf.size(); ++i) {
    (*predicted_cdf)[i] = (*predicted_cdf)[i - 1] + predicted_pdf[i];
  }
  return true;
}

bool Distribution::evaluate_one_instance_crps(
    const double& label_data,
    const std::vector<double>& predicted_dist,
    double* crps) {
  std::vector<double> predicted_cdf;
  pdf_to_cdf(predicted_dist, &predicted_cdf);
  double sum = pow(predicted_cdf[0], 2) * (m_target_bins_ptr_->at(1) - m_target_bins_ptr_->at(0));
  for (unsigned int i = 1; i < predicted_dist.size() - 1; ++i) {
    double tmp_crps = 0;
    if (label_data > m_target_bins_ptr_->at(i - 1)) {
      tmp_crps += pow(predicted_cdf[i], 2);
    } else {
      tmp_crps += pow(predicted_cdf[i] - 1.0, 2);
    }
    sum += tmp_crps * (m_target_bins_ptr_->at(i) - m_target_bins_ptr_->at(i - 1));
  }
  double target_bin_last_size =
      (m_target_bins_ptr_->at(m_target_bins_ptr_->size() - 1) -
      m_target_bins_ptr_->at(m_target_bins_ptr_->size() - 2));
  if (label_data > m_target_bins_ptr_->back()) {
    sum += pow(predicted_cdf[predicted_dist.size() - 1], 2) * target_bin_last_size;
  } else {
    sum += pow(predicted_cdf[predicted_dist.size() - 1] - 1.0, 2) * target_bin_last_size;
  }
  *crps = sum;
  return true;
}

bool Distribution::evaluate_crps(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& record_index_vec,
    const std::vector<std::vector<double>>& predicted_dist,
    double* crps) {
  double sum_crps = 0;
  for (unsigned long i = 0; i < record_index_vec.size(); ++i) {
    uint64_t record_index = record_index_vec[i];
    double tmp_crps;
    evaluate_one_instance_crps(label_data[record_index], predicted_dist[record_index], &tmp_crps);
    sum_crps += tmp_crps;
    // evaluate_one_instance_crps(label_data[]);
  }
  *crps = sum_crps;
  return true;
}

std::shared_ptr<Distribution> DistributionManager::get_distribution(PBTree_DistributionType type) {
  std::shared_ptr<Distribution> distribution_ptr;
  switch (type)
  {
  case PBTree_DistributionType_NORMAL_DISTRIBUTION:
    distribution_ptr = std::shared_ptr<Distribution>(new NormalDistribution());
    break;
  case PBTree_DistributionType_GAMMA_DISTRIBUTION:
    distribution_ptr = std::shared_ptr<Distribution>(new GammaDistribution());
    break;
  case PBTree_DistributionType_NONPARAMETRIC_CONTINUOUS:
    distribution_ptr = std::shared_ptr<Distribution>(new NonparametricContinousDistribution());
    break;
  case PBTree_DistributionType_BAYESIAN_CONTINUOUS:
    distribution_ptr = std::shared_ptr<Distribution>(new BayesianContinuousDistribution());
    break;
  default:
    LOG(FATAL) << "Unrecognized distribution type " << type;
    break;
  }
  return distribution_ptr;
}

}  // namespace pbtree
