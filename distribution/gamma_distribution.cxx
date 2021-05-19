#include "gamma_distribution.h"

DECLARE_double(min_prob);
DECLARE_double(regularization_param);
DECLARE_uint32(distribution_sample_point_num);
DEFINE_double(gamma_k_lower_bound, 0, "");

namespace pbtree {

bool GammaDistribution::transform_param(
    const double& raw_p1, const double& raw_p2, const double& raw_p3,
    double* p1, double* p2, double* p3) {
  *p1 = exp(raw_p1) + FLAGS_gamma_k_lower_bound;
  *p2 = exp(raw_p2);
  return true;
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
    const std::tuple<double, double, double>& param,
    double* first_moment, double* second_moment) {
  double k = std::get<0>(param);
  double theta = std::get<1>(param);
  *first_moment = k * theta;
  // if (k > 1) *first_moment = (k - 1) * theta;
  *second_moment = k * theta * theta;
  return true;
}

/**
 * @brief  
 * @note   
 * @param  label_data: 
 * @param  record_index_vec: 
 * @param  std::vector<std::tuple<double: 
 * @param  predicted_param: 
 * @param  g_p1: 
 * @param  g_p2: 
 * @param  g_p3: 
 * @retval 
 */
bool GammaDistribution::calculate_boost_gradient(
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

    double k, theta;
    transform_param(log_k, log_theta, 0, &k, &theta, nullptr);
    // double k = exp(log_k);
    // double theta = exp(log_theta);

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

/**
 * @brief  deprecated
 * @note   
 * @param  label_data: 
 * @param  record_index_vec: 
 * @param  std::vector<std::tuple<double: 
 * @param  predicted_param: 
 * @param  g_p1: 
 * @param  g_p2: 
 * @param  g_p3: 
 * @retval 
 */
bool _calculate_boost_gradient(
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
    calculate_boost_gradient(label_data, record_index_vec, predicted_param, &delta_k, &delta_theta, nullptr);
  }
  if (std::isnan(delta_k) || std::isnan(delta_theta)) {
    LOG(WARNING) << "Delta_k " << delta_k << " delta_theta " << delta_theta;
  }
  if (std::isnan(delta_k)) delta_k = 0;
  if (std::isnan(delta_theta)) delta_theta = 0;
  VLOG(101) << "Evaluation = " << evaluation << " Gradient delta_k = "
            << delta_k << ", gradient delta_theta = " << delta_theta;
  for (auto iter = record_index_vec.begin(); iter != record_index_vec.end(); ++iter) {
    auto param = predicted_param[*iter];
    double k = std::get<0>(param);
    double theta = std::get<1>(param);
    double new_k = k + delta_k;
    double new_theta = theta + delta_theta;
    double raw_k, raw_theta;
    transform_param(new_k, new_theta, 0, &raw_k, &raw_theta, nullptr);
    // double raw_k = exp(new_k);
    // double raw_theta = exp(new_theta);
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
  calculate_boost_gradient(label_data, record_index_vec, predicted_param, &delta_k, &delta_theta, nullptr);
  // if (Utility::check_double_le(delta_k, 0)) delta_k = 1e-3;
  // if (Utility::check_double_le(delta_theta, 0)) delta_theta = 1e-3;
  node->set_p1(delta_k);
  node->set_p2(delta_theta);
  node->set_distribution_type(PBTree_DistributionType_GAMMA_DISTRIBUTION);
  return true;
}

bool GammaDistribution::predict_interval(
    const double& p1, const double& p2, const double& p3,
    const double& lower_interval, const double& upper_interval,
    double* lower_bound, double* upper_bound) {
  boost::math::gamma_distribution<double> dist(p1, p2);
  *lower_bound = boost::math::quantile(dist, lower_interval);
  *upper_bound = boost::math::quantile(dist, upper_interval);
  return true;
}

// bool GammaDistribution::evaluate_rmsle(
//     const std::vector<double>& label_data,
//     const std::vector<uint64_t>& record_index_vec,
//     const std::vector<std::tuple<double, double, double>>& predicted_param,
//     double* rmsle) {
//   for (int i = 0; i < record_index_vec.size(); ++i)  {
//     uint64_t record_index = record_index_vec[i];
//     double first_moment = 0, second_moment = 0;
//     param_to_moment(predicted_param[record_index], &first_moment, &second_moment);
//     *rmsle += pow(log((label_data[record_index] + 1) / (first_moment + 1)), 2);
//   }
//   *rmsle = sqrt(*rmsle / record_index_vec.size());
//   return true;
// }

}  // namespace pbtree
