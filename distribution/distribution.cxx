#include "math.h"
#include "distribution.h"

DEFINE_uint32(distribution_sample_point_num, 100, "");
DEFINE_double(regularization_param, 0.1, "");
DEFINE_double(min_prob, 1e-100, "");
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
  node->set_distribution_type(PBTree_DistributionType_NORMAL_DISTRIBUITION);
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
    std::tuple<double, double, double>& param,
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
  LOG(FATAL) << "Not implemented yet";
  return true;
}

bool NormalDistribution::calculate_boost_loss(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& record_index_vec,
    const std::vector<std::tuple<double, double, double>>& predicted_param,
    double* loss,
    const bool& evaluation) {
  double mean = 0;
  double variance = 0;
  calc_sample_moment(label_data, record_index_vec, &mean, &variance);
  double mu_likelihood = mean;
  double sigma_likelihood = sqrt(variance);

  double tmp_loss = 0;
  for (auto iter = record_index_vec.begin(); iter != record_index_vec.end(); ++iter) {
    auto param = predicted_param[*iter];
    double mu_prior = std::get<0>(param);
    double sigma_prior = std::get<1>(param);
    // Refer to https://ccrma.stanford.edu/~jos/sasp/Product_Two_Gaussian_PDFs.html
    double mu_posterior = mu_prior * pow(sigma_likelihood, 2) +
        mu_likelihood * pow(sigma_prior, 2);
    double sigma_posterior = pow(sigma_likelihood, 2) * pow(sigma_prior, 2) /
        (pow(sigma_likelihood, 2) + pow(sigma_prior, 2));
    sigma_posterior = sqrt(sigma_posterior);
    boost::math::gamma_distribution<double> dist_posterior(mu_posterior, sigma_posterior);
    tmp_loss += log(boost::math::pdf(dist_posterior, label_data[*iter]));
  }
  *loss = tmp_loss * -1 / record_index_vec.size();
  return true;
}

std::shared_ptr<Distribution> DistributionManager::get_distribution(PBTree_DistributionType type) {
  std::shared_ptr<Distribution> distribution_ptr;
  switch (type)
  {
  case PBTree_DistributionType_NORMAL_DISTRIBUITION:
    distribution_ptr = std::shared_ptr<Distribution>(new NormalDistribution());
    break;
  case PBTree_DistributionType_GAMMA_DISTRIBUTION:
    distribution_ptr = std::shared_ptr<Distribution>(new GammaDistribution());
    break;
  default:
    LOG(FATAL) << "Unrecognized distribution type " << type;
    break;
  }
  return distribution_ptr;
}

bool GammaDistribution::calculate_loss(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& row_index_vec,
    double* loss, double* p1 /*= nullptr*/, double* p2 /*= nullptr*/, double* p3 /*= nullptr*/) {
  double mu = 0;
  // double sigma = 0;
  double square_sum = 0;
  // Compute mu and sigma
  for (auto iter = row_index_vec.begin(); iter < row_index_vec.end(); ++iter) {
    mu += label_data[*iter];
    square_sum += label_data[*iter] * label_data[*iter];
  }
  mu /= row_index_vec.size();
  // E(X^2) - (EX)^2
  double variance = square_sum / row_index_vec.size() - mu * mu;

  // For gamma(k, theta), mean = k * theta, variance = k * theta ^ 2
  // Therefore theta = variance / mean, k = mean / theta
  double theta = variance / mu;
  double k = mu / theta;
  if (p1 != nullptr) *p1 = k;
  if (p2 != nullptr) *p2 = theta;
  boost::math::gamma_distribution<double> dist(k, theta);

  double tmp_loss = 0;
  for (auto iter = row_index_vec.begin(); iter < row_index_vec.end(); ++iter) {
    tmp_loss += log(boost::math::pdf(dist, label_data[*iter]));
  }
  *loss = tmp_loss * -1 / row_index_vec.size();
  return true;
}

bool GammaDistribution::set_tree_node_param(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& row_index_vec,
    PBTree_Node* node) {
  double mu = 0;
  // double sigma = 0;
  double square_sum = 0;
  // Compute mu and sigma
  for (auto iter = row_index_vec.begin(); iter < row_index_vec.end(); ++iter) {
    mu += label_data[*iter];
    square_sum += label_data[*iter] * label_data[*iter];
  }
  mu /= row_index_vec.size();
  // E(X^2) - (EX)^2
  double variance = square_sum / row_index_vec.size() - mu * mu;

  // For gamma(k, theta), mean = k * theta, variance = k * theta ^ 2
  // Therefore theta = variance / mean, k = mean / theta
  double theta = variance / mu;
  double k = mu / theta;
  node->set_p1(k);
  node->set_p2(theta);
  node->set_distribution_type(PBTree_DistributionType_GAMMA_DISTRIBUTION);
  return true;
}

bool GammaDistribution::plot_distribution_curve(
    const double& p1, const double& p2,
    const double& p3,
    std::string* output_str) {
  boost::math::gamma_distribution<double> dist(p1, p2);
  
  const double lower_bound = 
      p1 * p2 - 5 * sqrt(p1 * p2 * p2) > 0 ? p1 * p2 - 5 * sqrt(p1 * p2 * p2) : 0;
  const double upper_bound = p1 * p2 + 5 * sqrt(p1 * p2 * p2);
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

bool GammaDistribution::calculate_moment(
    const PBTree_Node& node,
    double* first_moment,
    double* second_moment) {
  *first_moment = node.p1() * node.p2();
  *second_moment = node.p1() * node.p2() * node.p2();
  return true;
}

bool GammaDistribution::param_to_moment(
    std::tuple<double, double, double>& param,
    double* first_moment, double* second_moment) {
  double k = std::get<0>(param);
  double theta = std::get<1>(param);
  *first_moment = k * theta;
  *second_moment = k * theta * theta;
  return true;
}


bool calculate_boost_log_gradient(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& record_index_vec,
    const std::vector<std::tuple<double, double, double>>& predicted_param,
    double* g_p1, double* g_p2, double* g_p3) {
  if (g_p1 == nullptr || g_p2 == nullptr) {
    LOG(FATAL) << "Pointer for g_p1 or g_p2 is null";
    return false;
  }
  double sum_gradient_k = 0;
  double sum_gradient_theta = 0;
  /*
   * Implementation of formula solution
   * 
   * log(Gamma(y | k, theta)) = (k - 1) log(y) - y / theta - log(gamma(k)) - k * log(theta)
   * gradient_k = d log(Gamma(y | k, theta)) / d k = log(y / theta) - digamma(k)
   * gradient_gamma = d log(Gamma(y | k, theta)) / d theta = y / (theta^2) - k / theta
   * gradient_log_k = d log(Gamma(y | k, theta)) / d log_k = (log(y / theta) - digamma(k)) * e^log_k
   * gradient_log_gamma = d log(Gamma(y | k, theta)) / d log_theta = (y / (theta^2) - k / theta) * e^log_theta
   */

  for (auto iter = record_index_vec.begin(); iter != record_index_vec.end(); ++iter) {
    auto param = predicted_param[*iter];
    double log_k = std::get<0>(param);
    double log_theta = std::get<1>(param);

    double k = exp(log_k);
    double theta = exp(log_theta);

    double y = label_data[*iter];
    double gradient_log_k = (log(y / theta) - boost::math::digamma(k)) * k;
    double gradient_log_theta = (y / pow(theta, 2) - k / theta) * theta;
    sum_gradient_k += gradient_log_k;
    sum_gradient_theta += gradient_log_theta;
  }
  double gradient_k = sum_gradient_k * FLAGS_regularization_param;
  double gradient_theta = sum_gradient_theta * FLAGS_regularization_param;
  if (g_p1 != nullptr) *g_p1 = gradient_k / record_index_vec.size();
  if (g_p2 != nullptr) *g_p2 = gradient_theta / record_index_vec.size();
  return true;
}

bool GammaDistribution::calculate_boost_gradient(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& record_index_vec,
    const std::vector<std::tuple<double, double, double>>& predicted_param,
    double* g_p1, double* g_p2, double* g_p3) {
  if (g_p1 == nullptr || g_p2 == nullptr) {
    LOG(FATAL) << "Pointer for g_p1 or g_p2 is null";
    return false;
  }
  // double delta = 0.01;
  double sum_gradient_k = 0;
  double sum_gradient_theta = 0;
  /*
   * Implementation of formula solution
   * 
   * log(Gamma(y | k, theta)) = (k - 1) log(y) - y / theta - log(gamma(k)) - k * log(theta)
   * gradient_k = d log(Gamma(y | k, theta)) / d k = log(y / theta) - digamma(k)
   * gradient_gamma = d log(Gamma(y | k, theta)) / d theta = y / (theta^2) - k / theta
   * gradient_log_k = d log(Gamma(y | k, theta)) / d log_k = (log(y / theta) - digamma(k)) * e^log_k
   * gradient_log_gamma = d log(Gamma(y | k, theta)) / d log_theta = (y / (theta^2) - k / theta) * e^log_theta
   */

  for (auto iter = record_index_vec.begin(); iter != record_index_vec.end(); ++iter) {
    auto param = predicted_param[*iter];
    double k = std::get<0>(param);
    double theta = std::get<1>(param);
    double y = label_data[*iter];
    double gradient_k = log(y / theta) - boost::math::digamma(k);
    double gradient_theta = y / pow(theta, 2) - k / theta;
    sum_gradient_k += gradient_k;
    sum_gradient_theta += gradient_theta;
    // boost::math::gamma_distribution<double> dist_sample(k, theta);
    // boost::math::gamma_distribution<double> dist_delta_k(k + delta, theta);
    // boost::math::gamma_distribution<double> dist_delta_theta(k, theta + delta);
    // double p0 = boost::math::pdf(dist_sample, label_data[*iter]);
    // double p_k = boost::math::pdf(dist_delta_k, label_data[*iter]);
    // sum_gradient_k += log(p_k / p0) / delta;
    // double p_theta = boost::math::pdf(dist_delta_theta, label_data[*iter] + delta);
    // sum_gradient_theta += log(p_theta / p0) / delta;
  }
  double gradient_k = sum_gradient_k * FLAGS_regularization_param;
  double gradient_theta = sum_gradient_theta * FLAGS_regularization_param;
  if (g_p1 != nullptr) *g_p1 = gradient_k / record_index_vec.size();
  if (g_p2 != nullptr) *g_p2 = gradient_theta / record_index_vec.size();
  return true;
}

bool GammaDistribution::calculate_boost_loss(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& record_index_vec,
    const std::vector<std::tuple<double, double, double>>& predicted_param,
    double* loss,
    const bool& evaluation) {
  double tmp_loss = 0;
  double delta_k = 0, delta_theta = 0;
  if (!evaluation) {
    calculate_boost_log_gradient(label_data, record_index_vec, predicted_param, &delta_k, &delta_theta, nullptr);
  }
  if (std::isnan(delta_k) || std::isnan(delta_theta)) {
    LOG(WARNING) << "Delta_k " << delta_k << " delta_theta " << delta_theta;
  }
  if (std::isnan(delta_k)) delta_k = 0;
  if (std::isnan(delta_theta)) delta_theta = 0;
  LOG(INFO) << "Evaluation = " << evaluation << " Gradient delta_k = "
            << delta_k << ", gradient delta_theta = " << delta_theta;
  for (auto iter = record_index_vec.begin(); iter != record_index_vec.end(); ++iter) {
    auto param = predicted_param[*iter];
    double k = std::get<0>(param);
    double theta = std::get<1>(param);
    double new_k = k + delta_k;
    double new_theta = theta + delta_theta;
    double raw_k = exp(new_k);
    double raw_theta = exp(new_theta);
    // if (Utility::check_double_le(new_k, 0) || std::isnan(new_k)) new_k = 1e-3;
    // if (Utility::check_double_le(new_theta, 0) || std::isnan(new_theta)) new_theta = 1e-3;
    boost::math::gamma_distribution<double> dist_sample(raw_k, raw_theta);
    double prob = boost::math::pdf(dist_sample, label_data[*iter]);
    if (prob < FLAGS_min_prob) {
      prob = FLAGS_min_prob;
    }
    double log_loss = log(prob);
    if (std::isinf(prob) || std::isinf(log_loss)) {
      LOG(WARNING) << "Raw_k = " << raw_k << ", raw_theta = " << raw_theta
                   << ", label = " << label_data[*iter]
                   << ", prob = " << prob << ", log_loss = " << log_loss;
    }
    tmp_loss += log_loss;
  }
  *loss = tmp_loss * -1 / record_index_vec.size();
  return true;
}

bool GammaDistribution::set_boost_node_param(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& record_index_vec,
    const std::vector<std::tuple<double, double, double>>& predicted_param,
    PBTree_Node* node) {
  double delta_k = 0, delta_theta = 0;
  calculate_boost_log_gradient(label_data, record_index_vec, predicted_param, &delta_k, &delta_theta, nullptr);
  // if (Utility::check_double_le(delta_k, 0)) delta_k = 1e-3;
  // if (Utility::check_double_le(delta_theta, 0)) delta_theta = 1e-3;
  node->set_p1(delta_k);
  node->set_p2(delta_theta);
  node->set_distribution_type(PBTree_DistributionType_GAMMA_DISTRIBUTION);
  return true;
}
// bool GammaDistribution::calculate_boost_loss(
//     const std::vector<double>& label_data,
//     const std::vector<uint64_t>& record_index_vec,
//     const std::vector<std::tuple<double, double, double>>& predicted_param,
//     double* loss) {
//   double mean = 0;
//   double variance = 0;
//   calc_sample_moment(label_data, record_index_vec, &mean, &variance);
//   double theta_likelihood = variance / mean;
//   double k_likelihood = mean / theta_likelihood;
//   double tmp_loss = 0;

//   for (auto iter = record_index_vec.begin(); iter != record_index_vec.end(); ++iter) {
//     auto param = predicted_param[*iter];
//     double k_prior = std::get<0>(param);
//     double theta_prior = std::get<1>(param);
//     // k = k1 + k2 - 1, theta = 1 / (1 / theta1 + 1 / theta2)
//     double k_posterior = k_prior + k_likelihood - 1;
//     double theta_posterior = 1.0 / ( 1 / theta_prior + 1 / theta_likelihood);
//     boost::math::gamma_distribution<double> dist_posterior(k_posterior, theta_posterior);
//     tmp_loss += log(boost::math::pdf(dist_posterior, label_data[*iter]));
//   }
//   *loss = tmp_loss * -1 / record_index_vec.size();
//   return true;
// }

}  // pbtree
