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
    const std::vector<double>& raw_dist,
    std::vector<double>* pred_dist) {
  pred_dist->resize(2);
  (*pred_dist)[0] = raw_dist[0];
  (*pred_dist)[1] = exp(raw_dist[1]);
  return true;
}

bool NormalDistribution::calculate_loss(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& row_index_vec,
    double* loss, std::vector<double>* distribution /*= nullptr*/) {
  double mu = 0;
  double sigma = 0;

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

  *loss = tmp_loss * -1 / row_index_vec.size();
  if (distribution) {
    distribution->resize(2);
    (*distribution)[0] = mu;
    (*distribution)[1] = sigma;
  }
  return true;
}

bool NormalDistribution::set_boost_node_param(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& row_index_vec,
    const std::vector<std::vector<double>>& prior,
    PBTree_Node* node) {
  std::vector<double> likelihood;
  calculate_boost_gradient(label_data, row_index_vec, prior, &likelihood);
  double delta_mu = likelihood[0], delta_sigma = likelihood[1];
  node->set_p1(delta_mu);
  node->set_p2(delta_sigma);
  node->add_target_dist(delta_mu);
  node->add_target_dist(delta_sigma);
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
    const std::vector<double>& distribution,
    std::string* output_str) {
  double p1 = distribution[0], p2 = distribution[1];
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
    const std::vector<double>& distribution,
    double* first_moment, double* second_moment) {
  double mu = distribution[0];
  double sigma = distribution[1];
  *first_moment = mu;
  *second_moment = sigma * sigma;
  return true;
}

bool NormalDistribution::calculate_boost_gradient(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& record_index_vec,
    const std::vector<std::vector<double>>& prior,
    std::vector<double>* likelihood) {
  if (likelihood == nullptr) {
    LOG(FATAL) << "Pointer for likelihood is null";
    return false;
  }
  double sum_gradient_mu = 0;
  double sum_gradient_sigma = 0;
  for (auto iter = record_index_vec.begin(); iter != record_index_vec.end(); ++iter) {
    auto param = prior[*iter];
    // double raw_mu = std::get<0>(param);
    // double raw_sigma = std::get<1>(param);
    std::vector<double> transformed_param;
    transform_param(param, &transformed_param);
    double mu = transformed_param[0];
    double sigma = transformed_param[1];
    double y = label_data[*iter];
    double gradient_mu = (y - mu) / pow(sigma, 2);
    double gradient_log_sigma = -1 + pow(y - mu, 2) * pow(sigma, -2)
        - pow(sigma, 2) * FLAGS_regularization_param2;
    sum_gradient_mu += gradient_mu;
    sum_gradient_sigma += gradient_log_sigma;
  }
  double gradient_mu = sum_gradient_mu * FLAGS_learning_rate1;
  double gradient_sigma = sum_gradient_sigma * FLAGS_learning_rate2;
  // if(FLAGS_learning_rate2 > 0.0001)
  //   LOG(INFO) << "sum_gradient_sigma = " << sum_gradient_sigma;
  likelihood->resize(2);
  (*likelihood)[0] = gradient_mu / record_index_vec.size();
  (*likelihood)[1] = gradient_sigma / record_index_vec.size();
  return true;
}

bool NormalDistribution::calculate_boost_loss(
    const std::vector<double>& label_data,
    const std::vector<uint64_t>& record_index_vec,
    const std::vector<std::vector<double>>& prior,
    double* loss,
    const bool& evaluation) {
  double tmp_loss = 0;
  double delta_mu = 0, delta_sigma = 0;
  std::vector<double> likelihood;
  if (!evaluation) {
    calculate_boost_gradient(label_data, record_index_vec, prior, &likelihood);
    delta_mu = likelihood[0];
    delta_sigma = likelihood[1];
  }
  if (std::isnan(delta_mu) || std::isnan(delta_sigma)) {
    LOG(WARNING) << "Delta_mu " << delta_mu << " delta_sigma " << delta_sigma;
  }
  if (std::isnan(delta_mu)) delta_mu = 0;
  if (std::isnan(delta_sigma)) delta_sigma = 0;
  VLOG(101) << "Evaluation = " << evaluation << " Gradient delta_k = "
            << delta_mu << ", gradient delta_theta = " << delta_sigma;
  for (auto iter = record_index_vec.begin(); iter != record_index_vec.end(); ++iter) {
    auto param = prior[*iter];
    double old_mu = param[0];
    double old_sigma = param[1];
    double new_mu = old_mu + delta_mu;
    double new_sigma = old_sigma + delta_sigma;
    double mu, sigma;
    std::vector<double> raw_param = {new_mu, new_sigma};
    std::vector<double> transformed_param;
    transform_param(raw_param, &transformed_param);
    mu = transformed_param[0];
    sigma = transformed_param[1];
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
    const std::vector<double>& distribution,
    const double& lower_interval, const double& upper_interval,
    double* lower_bound, double* upper_bound) {
  double mu = distribution[0];
  double sigma = distribution[1];
  boost::math::normal_distribution<double> dist(mu, sigma);
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

}  // namespace pbtree
