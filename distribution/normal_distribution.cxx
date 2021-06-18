#include "normal_distribution.h"

DECLARE_double(min_prob);
DECLARE_double(regularization_param1);
DECLARE_double(regularization_param2);
DECLARE_double(learning_rate1);
DECLARE_double(learning_rate2);
DECLARE_uint32(distribution_sample_point_num);
DEFINE_uint64(gauss_alter_round, 10, "");

namespace pbtree {

bool NormalDistribution::transform_param(
    const double& raw_p1, const double& raw_p2, const double& raw_p3,
    double* p1, double* p2, double* p3) {
  *p1 = raw_p1;
  *p2 = exp(raw_p2);
  return true;
}

bool NormalDistribution::calculate_loss(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& row_index_vec,
    double* loss, double* p1 /*= nullptr*/, double* p2 /*= nullptr*/, double* p3 /*= nullptr*/) {
  double mu = 0;
  double sigma = 0;
  // double square_sum = 0;
  // // Compute mu and sigma
  // for (auto iter = row_index_vec.begin(); iter < row_index_vec.end(); ++iter) {
  //   mu += label_data[*iter];
  //   square_sum += label_data[*iter] * label_data[*iter];
  // }
  // mu /= row_index_vec.size();
  // E(X^2) - (EX)^2
  // double variance = square_sum / row_index_vec.size() - mu * mu;
  double variance = 0;
  calc_sample_moment(label_data, row_index_vec, &mu, &variance);
  if (!Utility::check_double_le(0, variance)) {
    sigma = -1;
  } else if (Utility::check_double_equal(0, variance)) {
    sigma = 0;
  } else {
    sigma = sqrt(variance);
  }
  // VLOG(102) << sigma;
  // Compute loss
  double tmp_loss = 0;
  boost::math::normal_distribution<double> dist(mu, sigma);
  for (auto iter = row_index_vec.begin(); iter < row_index_vec.end(); ++iter) {
    tmp_loss += log(boost::math::pdf(dist, label_data[*iter]));
  }
  // for (auto iter = row_index_vec.begin(); iter < row_index_vec.end(); ++iter) {
  //   tmp_loss += (mu - label_data[*iter]) * (mu - label_data[*iter]);
  // }
  *loss = tmp_loss * -1 / row_index_vec.size();
  if (p1 != nullptr) *p1 = mu;
  if (p2 != nullptr) *p2 = sigma;
  return true;
}

bool NormalDistribution::set_boost_node_param(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& row_index_vec,
    const std::vector<std::tuple<double, double, double>>& predicted_param,
    PBTree_Node* node) {
  double delta_mu = 0, delta_sigma = 0;
  calculate_boost_gradient(label_data, row_index_vec, predicted_param, &delta_mu, &delta_sigma, nullptr);
  // if (Utility::check_double_le(delta_k, 0)) delta_k = 1e-3;
  // if (Utility::check_double_le(delta_theta, 0)) delta_theta = 1e-3;
  node->set_p1(delta_mu);
  node->set_p2(delta_sigma);
  node->set_distribution_type(PBTree_DistributionType_NORMAL_DISTRIBUTION);
  return true;
}

bool NormalDistribution::set_tree_node_param(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& row_index_vec,
    PBTree_Node* node) {
  double mu = 0;
  double sigma = 0;
  double square_sum = 0;
  // Compute mu and sigma
  for (auto iter = row_index_vec.begin(); iter < row_index_vec.end(); ++iter) {
    mu += label_data[*iter];
    square_sum += label_data[*iter] * label_data[*iter];
  }
  mu /= row_index_vec.size();
  // E(X^2) - (EX)^2
  double variance = square_sum / row_index_vec.size() - mu * mu;
  if (!Utility::check_double_le(0, variance)) {
    sigma = -1;
  } else if (Utility::check_double_equal(0, variance)) {
    sigma = 0;
  } else {
    sigma = sqrt(variance);
  }
  sigma = sqrt(variance);
  node->set_p1(mu);
  node->set_p2(sigma);
  node->set_distribution_type(PBTree_DistributionType_NORMAL_DISTRIBUTION);
  return true;
}

bool NormalDistribution::plot_distribution_curve(
    const double& p1, const double& p2,
    const double& p3,
    std::string* output_str) {
  boost::math::normal_distribution<double> dist(p1, p2);
  
  const double lower_bound = p1 - 5 * p2;
  const double upper_bound = p1 + 5 * p2;
  const double step = (upper_bound - lower_bound) / FLAGS_distribution_sample_point_num;
  std::stringstream ss;
  for (unsigned int i = 0; i < FLAGS_distribution_sample_point_num; ++i) {
    double x = i * step + lower_bound;
    double y = boost::math::pdf(dist, x);
    ss << x << " " << y << "\n";
  }
  *output_str = ss.str();
  return true;
}

bool NormalDistribution::calculate_moment(
    const PBTree_Node& node,
    double* first_moment,
    double* second_moment) {
  *first_moment = node.p1();
  *second_moment = node.p2() * node.p2();
  return true;
}

bool NormalDistribution::param_to_moment(
    const std::tuple<double, double, double>& param,
    double* first_moment, double* second_moment) {
  double mu = std::get<0>(param);
  double sigma = std::get<1>(param);
  *first_moment = mu;
  *second_moment = sigma * sigma;
  return true;
}

bool NormalDistribution::calculate_boost_gradient(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& record_index_vec,
    const std::vector<std::tuple<double, double, double>>& predicted_param,
    double* g_p1, double* g_p2, double* g_p3) {
  if (g_p1 == nullptr || g_p2 == nullptr) {
    LOG(FATAL) << "Pointer for g_p1 or g_p2 is null";
    return false;
  }
  double sum_gradient_mu = 0;
  double sum_gradient_sigma = 0;
  for (auto iter = record_index_vec.begin(); iter != record_index_vec.end(); ++iter) {
    auto param = predicted_param[*iter];
    double raw_mu = std::get<0>(param);
    double raw_sigma = std::get<1>(param);
    double mu, sigma;
    transform_param(raw_mu, raw_sigma, 0, &mu, &sigma, nullptr);
    double y = label_data[*iter];
    double gradient_mu = (y - mu) / pow(sigma, 2);
    double gradient_log_sigma = -1 + pow(y - mu, 2) * pow(sigma, -2)
        - pow(sigma, 2) * FLAGS_regularization_param2;
    sum_gradient_mu += gradient_mu;
    sum_gradient_sigma += gradient_log_sigma;
  }
  double gradient_mu = sum_gradient_mu * FLAGS_learning_rate1;
  double gradient_sigma = sum_gradient_sigma * FLAGS_learning_rate2;
  if(FLAGS_learning_rate2 > 0.0001)
    LOG(INFO) << "sum_gradient_sigma = " << sum_gradient_sigma;
  if (g_p1 != nullptr) *g_p1 = gradient_mu / record_index_vec.size();
  if (g_p2 != nullptr) *g_p2 = gradient_sigma / record_index_vec.size();
  return true;
}

// bool NormalDistribution::calculate_boost_loss(
//     const std::vector<double>& label_data,
//     const std::vector<uint64_t>& record_index_vec,
//     const std::vector<std::tuple<double, double, double>>& predicted_param,
//     double* loss,
//     const bool& evaluation) {
//   double mean = 0;
//   double variance = 0;
//   calc_sample_moment(label_data, record_index_vec, &mean, &variance);
//   double mu_likelihood = mean;
//   double sigma_likelihood = sqrt(variance);

//   double tmp_loss = 0;
//   for (auto iter = record_index_vec.begin(); iter != record_index_vec.end(); ++iter) {
//     auto param = predicted_param[*iter];
//     double mu_prior = std::get<0>(param);
//     double sigma_prior = std::get<1>(param);
//     // Refer to https://ccrma.stanford.edu/~jos/sasp/Product_Two_Gaussian_PDFs.html
//     double mu_posterior = mu_prior * pow(sigma_likelihood, 2) +
//         mu_likelihood * pow(sigma_prior, 2);
//     double sigma_posterior = pow(sigma_likelihood, 2) * pow(sigma_prior, 2) /
//         (pow(sigma_likelihood, 2) + pow(sigma_prior, 2));
//     sigma_posterior = sqrt(sigma_posterior);
//     boost::math::gamma_distribution<double> dist_posterior(mu_posterior, sigma_posterior);
//     tmp_loss += log(boost::math::pdf(dist_posterior, label_data[*iter]));
//   }
//   *loss = tmp_loss * -1 / record_index_vec.size();
//   return true;
// }

bool NormalDistribution::calculate_boost_loss(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& record_index_vec,
    const std::vector<std::tuple<double, double, double>>& predicted_param,
    double* loss,
    const bool& evaluation) {
  double tmp_loss = 0;
  double delta_mu = 0, delta_sigma = 0;
  if (!evaluation) {
    calculate_boost_gradient(label_data, record_index_vec, predicted_param, &delta_mu, &delta_sigma, nullptr);
  }
  if (std::isnan(delta_mu) || std::isnan(delta_sigma)) {
    LOG(WARNING) << "Delta_mu " << delta_mu << " delta_sigma " << delta_sigma;
  }
  if (std::isnan(delta_mu)) delta_mu = 0;
  if (std::isnan(delta_sigma)) delta_sigma = 0;
  VLOG(101) << "Evaluation = " << evaluation << " Gradient delta_k = "
            << delta_mu << ", gradient delta_theta = " << delta_sigma;
  for (auto iter = record_index_vec.begin(); iter != record_index_vec.end(); ++iter) {
    auto param = predicted_param[*iter];
    double old_mu = std::get<0>(param);
    double old_sigma = std::get<1>(param);
    double new_mu = old_mu + delta_mu;
    double new_sigma = old_sigma + delta_sigma;
    double mu, sigma;
    transform_param(new_mu, new_sigma, 0, &mu, &sigma, nullptr);
    // double raw_k = exp(new_k);
    // double raw_theta = exp(new_theta);
    // if (Utility::check_double_le(new_k, 0) || std::isnan(new_k)) new_k = 1e-3;
    // if (Utility::check_double_le(new_theta, 0) || std::isnan(new_theta)) new_theta = 1e-3;
    boost::math::normal_distribution<double> dist_sample(mu, sigma);
    double prob = boost::math::pdf(dist_sample, label_data[*iter]);
    if (prob < FLAGS_min_prob) {
      prob = FLAGS_min_prob;
    }
    double log_loss = log(prob)
        - 0.5 * FLAGS_regularization_param2 * pow(sigma, 2);
    if (std::isinf(prob) || std::isinf(log_loss)) {
      LOG(WARNING) << "Raw_mu = " << mu << ", raw_sigma = " << sigma
                   << ", label = " << label_data[*iter]
                   << ", prob = " << prob << ", log_loss = " << log_loss;
    }
    tmp_loss += log_loss;
  }
  *loss = tmp_loss * -1 / record_index_vec.size();
  return true;
}

bool NormalDistribution::predict_interval(
    const double& p1, const double& p2, const double& p3,
    const double& lower_interval, const double& upper_interval,
    double* lower_bound, double* upper_bound) {
  boost::math::normal_distribution<double> dist(p1, p2);
  *lower_bound = boost::math::quantile(dist, lower_interval);
  *upper_bound = boost::math::quantile(dist, upper_interval);
  return true;
}

bool NormalDistribution::get_learning_rate(
    const uint64_t& round,
    const double& initial_p1_learning_rate,
    const double& initial_p2_learning_rate,
    const double& initial_p3_learning_rate,
    double* p1_learning_rate,
    double* p2_learning_rate, double* p3_learning_rate) {
  if (round < FLAGS_gauss_alter_round) {
    *p1_learning_rate = initial_p1_learning_rate;
    *p2_learning_rate = 0;
  } else {
    *p1_learning_rate = initial_p1_learning_rate;
    *p2_learning_rate = initial_p2_learning_rate;
  }
  return true;
}

// bool NormalDistribution::evaluate_rmsle(
//     const std::vector<double>& label_data,
//     const std::vector<uint64_t>& record_index_vec,
//     const std::vector<std::tuple<double, double, double>>& predicted_param,
//     double* rmsle) {
//   LOG(FATAL) << "Not implemented yet";
//   return true;
// }

}  // namespace pbtree
