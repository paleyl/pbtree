#include "gamma_distribution.h"

DECLARE_double(min_prob);
DECLARE_double(min_log_prob);
DECLARE_double(regularization_param1);
DECLARE_double(regularization_param2);
DECLARE_double(learning_rate1);
DECLARE_double(learning_rate2);
DECLARE_uint32(distribution_sample_point_num);
DECLARE_double(crps_evaluate_bins);
DEFINE_double(gamma_k_lower_bound, 0, "");
DEFINE_double(gamma_init_p1, 2.0, "");
DEFINE_double(gamma_init_p2, 1.0, "");
DEFINE_uint64(gamma_alter_round, 10, "");

namespace pbtree {

bool GammaDistribution::init_param(std::vector<double>* init_dist) {
  init_dist->resize(2);
  (*init_dist)[0] = FLAGS_gamma_init_p1;  // log_k
  (*init_dist)[1] = FLAGS_gamma_init_p2;  // log_theta
  return true;
}

bool GammaDistribution::transform_param(
      const std::vector<double>& raw_dist,
      std::vector<double>* transformed_dist) {
  transformed_dist->resize(2);
  (*transformed_dist)[0] = exp(raw_dist[0]);
  (*transformed_dist)[1] = exp(raw_dist[1]);
  return true;
}

bool GammaDistribution::calculate_loss(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& row_index_vec,
    double* loss, std::vector<double>* distribution /*= nullptr*/) {
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
  if (distribution) {
    distribution->resize(2);
    (*distribution)[0] = k;
    (*distribution)[1] = theta;
  }
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
    const std::vector<double>& distribution,
    std::string* output_str) {
  double k = distribution[0];
  double theta = distribution[1];
  boost::math::gamma_distribution<double> dist(k, theta);
  
  const double lower_bound = 
      k * theta - 5 * sqrt(k * theta * theta) > 0 ? k * theta - 5 * sqrt(k * theta * theta) : 0;
  const double upper_bound = k * theta + 5 * sqrt(k * theta * theta);
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
  double k = node.target_dist(0);
  double theta = node.target_dist(1);
  *first_moment = k * theta;
  *second_moment = k * theta * theta;
  return true;
}

bool GammaDistribution::param_to_moment(
    const std::vector<double>& distribution,
    double* first_moment, double* second_moment) {
  double k = distribution[0];
  double theta = distribution[1];
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
 * @param  prior: 
 * @param  likelihood: 
 * @retval 
 */
bool GammaDistribution::calculate_boost_gradient(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& record_index_vec,
    const std::vector<std::vector<double>>& prior,
    std::vector<double>* likelihood) {
  if (likelihood == nullptr) {
    LOG(FATAL) << "Pointer for likelihood vec is null";
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
    auto param = prior[*iter];
    std::vector<double> transformed_param;
    transform_param(param, &transformed_param);
    double k = transformed_param[0];
    double theta = transformed_param[1];

    double y = label_data[*iter];
    double gradient_log_k = 
       (log(y / theta) - boost::math::digamma(k)) * k
        - 2 * FLAGS_regularization_param1 * log(k / FLAGS_regularization_param2)
        ;
    double gradient_log_theta = (y / pow(theta, 2) - k / theta) * theta;
    sum_gradient_k += gradient_log_k;
    sum_gradient_theta += gradient_log_theta;
  }
  double gradient_k = sum_gradient_k * FLAGS_learning_rate1;
  double gradient_theta = sum_gradient_theta * FLAGS_learning_rate2;
  likelihood->resize(2);
  (*likelihood)[0] = gradient_k / record_index_vec.size();
  (*likelihood)[1] = gradient_theta / record_index_vec.size();
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
  }
  double gradient_k = sum_gradient_k * FLAGS_learning_rate1;
  double gradient_theta = sum_gradient_theta * FLAGS_learning_rate2;
  if (g_p1 != nullptr) *g_p1 = gradient_k / record_index_vec.size();
  if (g_p2 != nullptr) *g_p2 = gradient_theta / record_index_vec.size();
  return true;
}

bool GammaDistribution::calculate_boost_loss(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& record_index_vec,
    const std::vector<std::vector<double>>& prior,
    double* loss,
    const bool& evaluation) {
  double tmp_loss = 0;
  double delta_k = 0, delta_theta = 0;
  std::vector<double> likelihood;
  if (!evaluation) {
    calculate_boost_gradient(label_data, record_index_vec, prior, &likelihood);
    delta_k = likelihood[0];
    delta_theta = likelihood[1];
  }
  if (std::isnan(delta_k) || std::isnan(delta_theta)) {
    LOG(WARNING) << "Delta_k " << delta_k << " delta_theta " << delta_theta;
  }
  if (std::isnan(delta_k)) delta_k = 0;
  if (std::isnan(delta_theta)) delta_theta = 0;
  VLOG(101) << "Evaluation = " << evaluation << " Gradient delta_k = "
            << delta_k << ", gradient delta_theta = " << delta_theta;
  for (auto iter = record_index_vec.begin(); iter != record_index_vec.end(); ++iter) {
    auto param = prior[*iter];
    double k = param[0];
    double theta = param[1];
    double new_k = k + delta_k;
    double new_theta = theta + delta_theta;
    std::vector<double> new_dist = {new_k, new_theta};
    std::vector<double> raw_dist;
    transform_param(new_dist, &raw_dist);
    double one_instance_loss = 0;
    evaluate_one_instance_loss(label_data[*iter], raw_dist, &one_instance_loss);
    // double raw_k = raw_dist[0], raw_theta = raw_dist[1];
    // boost::math::gamma_distribution<double> dist_sample(raw_k, raw_theta);
    // double prob = boost::math::pdf(dist_sample, label_data[*iter]);
    // if (prob < FLAGS_min_prob) {
    //   prob = FLAGS_min_prob;
    // }
    // double log_loss = log(prob)
    //     - FLAGS_regularization_param1 * pow(log(raw_k / FLAGS_regularization_param2), 2);
    // if (std::isinf(prob) || std::isinf(log_loss)) {
    //   LOG(WARNING) << "Raw_k = " << raw_k << ", raw_theta = " << raw_theta
    //                << ", label = " << label_data[*iter]
    //                << ", prob = " << prob << ", log_loss = " << log_loss;
    // }
    tmp_loss += one_instance_loss;
  }
  *loss = tmp_loss * -1 / record_index_vec.size();
  return true;
}

bool GammaDistribution::set_boost_node_param(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& record_index_vec,
    const std::vector<std::vector<double>>& prior,
    PBTree_Node* node) {
//  double delta_k = 0, delta_theta = 0;
  std::vector<double> likelihood;
  calculate_boost_gradient(label_data, record_index_vec, prior, &likelihood);
  double delta_k = likelihood[0];
  double delta_theta = likelihood[1];
  node->set_p1(delta_k);
  node->set_p2(delta_theta);
  node->clear_target_dist();
  node->add_target_dist(delta_k);
  node->add_target_dist(delta_theta);
  node->set_distribution_type(PBTree_DistributionType_GAMMA_DISTRIBUTION);
  return true;
}

bool GammaDistribution::predict_interval(
    const std::vector<double>& distribution,
    const double& lower_interval, const double& upper_interval,
    double* lower_bound, double* upper_bound) {
  double k = distribution[0];
  double theta = distribution[1];
  boost::math::gamma_distribution<double> dist(k, theta);
  *lower_bound = boost::math::quantile(dist, lower_interval);
  *upper_bound = boost::math::quantile(dist, upper_interval);
  return true;
}

bool GammaDistribution::get_learning_rate(
    const uint64_t& round,
    const double& initial_p1_learning_rate,
    const double& initial_p2_learning_rate,
    const double& initial_p3_learning_rate,
    double* p1_learning_rate,
    double* p2_learning_rate, double* p3_learning_rate) {

  if (round < FLAGS_gamma_alter_round) {
    *p1_learning_rate = 0;
    *p2_learning_rate = initial_p2_learning_rate;
  } else {
    *p1_learning_rate = initial_p1_learning_rate;
    *p2_learning_rate = initial_p2_learning_rate;
  }
  return true;
}

bool GammaDistribution::evaluate_crps(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& record_index_vec,
    const std::vector<std::vector<double>>& predicted_dist,
    double* crps) {
  *crps = 0;
  for (unsigned int i = 0; i < record_index_vec.size(); ++i) {
    uint64_t record_index = record_index_vec[i];
    std::vector<double> transformed_dist;
    transform_param(predicted_dist[record_index],
      &transformed_dist);
    double one_instance_crps;
    evaluate_one_instance_crps(
        label_data[record_index], transformed_dist, &one_instance_crps);
    if (one_instance_crps > 10000) {
      LOG(WARNING) << "large crps: label = " << label_data[record_index] << ", (" << transformed_dist[0]
                << "," << transformed_dist[1] << "), " << one_instance_crps;
    }
    // LOG_EVERY_N(INFO, 1000) << "label = " << label_data[record_index] << ", (" << transformed_dist[0]
    //                         << "," << transformed_dist[1] << "), " << one_instance_crps;
    *crps += one_instance_crps;
  }
  *crps /= record_index_vec.size();
  return true;
}

bool GammaDistribution::evaluate_one_instance_crps(
    const double& label_data,
    const std::vector<double>& predicted_dist,
    double* crps) {
  double k = predicted_dist[0];
  double theta = predicted_dist[1];
  boost::math::gamma_distribution<double> dist(k, theta);
  *crps = 0;
  double pre_x = 0;
  // double rand_num = rand() * 1.0 / RAND_MAX;
  for (unsigned int i = 1; i < FLAGS_crps_evaluate_bins; ++i) {
    double q = 1.0 / FLAGS_crps_evaluate_bins * i;
    double x = boost::math::quantile(dist, q);
    double cdf_diff = x > label_data ? 1.0 - q : q;
    double tmp_crps = (pow(cdf_diff, 2) * (x - pre_x));
    *crps += tmp_crps;
    // if (rand_num < 0.0001) {
    //   std::stringstream ss;
    //   ss << label_data << " (" << k << "," << theta << ") "
    //      << q << " " << x << " " << pre_x << " " << tmp_crps << " " << *crps;
    //   LOG(INFO) << ss.str();
    // }
    pre_x = x;
  }
  return true;      
}

bool GammaDistribution::evaluate_logp(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& record_index_vec,
    const std::vector<std::vector<double>>& predicted_dist,
    double* logp) {
  *logp = 0;
  for (unsigned int i = 0; i < record_index_vec.size(); ++i) {
    uint64_t record_index = record_index_vec[i];
    std::vector<double> transformed_dist;
    transform_param(predicted_dist[record_index],
      &transformed_dist);
    double one_instance_logp;
    evaluate_one_instance_logp(
        label_data[record_index], predicted_dist[record_index], &one_instance_logp);
    *logp += one_instance_logp;
  }
  *logp /= record_index_vec.size();
  return true;
}

bool GammaDistribution::evaluate_one_instance_logp(
    const double& label_data,
    const std::vector<double>& predicted_dist,
    double* logp) {
  double k = predicted_dist[0];
  double theta = predicted_dist[1];
  // Calculate gamma logp from pdf
  double log_prob = - log(boost::math::tgamma(k)) - k * log(theta) + (k - 1) * log(label_data) - label_data / theta;
  // boost::math::gamma_distribution<double> dist(k, theta);
  // double prob = boost::math::pdf(dist, label_data);
  // double log_prob = log(prob);
  // if (prob < FLAGS_min_prob) {
  //   log_prob = FLAGS_min_log_prob;
  // }
  *logp = log_prob - FLAGS_regularization_param1 * pow(log(k / FLAGS_regularization_param2), 2);
  // LOG_EVERY_N(INFO, 100000) << "Evaluating gamma logp, label = " << label_data << ", logp = " << *logp;
  return true;
}

}  // namespace pbtree
