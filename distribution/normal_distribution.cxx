#include "normal_distribution.h"

DECLARE_double(min_prob);
DECLARE_double(regularization_param);
DECLARE_uint32(distribution_sample_point_num);

namespace pbtree {

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

}  // namespace pbtree
