#include <sys/sysinfo.h>
#include <stdlib.h>

#include <chrono> 
#include "boost/asio/thread_pool.hpp"
#include "boost/asio/post.hpp"
#include "tree.h"

DEFINE_uint32(histogram_bin_count, 100, "Number of bins in histogram");
DEFINE_uint32(split_min_count, 5, "Minimum record count for split");
DEFINE_double(split_min_ratio, 0.02, "Minimum record count for split");
DEFINE_uint32(tree_max_depth, 10, "Minimum record count for split");
DEFINE_bool(boosting_mode, true, "");
DEFINE_uint32(training_round, 10, "");
DEFINE_double(split_gain_min_ratio, 0.01, "");
DEFINE_double(split_gain_min_value, 1e-5, "");
DEFINE_uint32(thread_num, 8, "");
DEFINE_uint32(save_temp_model_round, 1000, "");
DEFINE_string(output_model_path, "", "");
DEFINE_double(train_record_sample_ratio, 1.0, "");
DEFINE_double(lower_confidence_interval, 0.2, "");
DEFINE_double(upper_confidence_interval, 0.8, "");
DEFINE_double(feature_non_zero_convert_ratio, 0.003, "");
DEFINE_bool(evaluate_loss_every_node, false, "");
DEFINE_double(record_min_ratio, 5e-3, "");
DEFINE_uint32(filter_feature_batch_size, 1000, "");
DEFINE_bool(use_multi_thread_filter, false, "");
DEFINE_uint32(alter_coord_round, 5, "");
DECLARE_double(learning_rate1);
DECLARE_double(learning_rate2);
DEFINE_string(burned_model_path, "", "");

// TODO(paleylv): current thread pool is not stable, 
// use_multi_thread_filter should be false to avoid crash
// use boost thread pool to replace current version.
namespace pbtree {

bool Tree::init() {
  m_max_non_zero_per_feature_ = 0;
  unsigned int num_cpu_cores = get_nprocs_conf();
  int thread_num = num_cpu_cores * 2 / 3 < FLAGS_thread_num ?
      num_cpu_cores * 2 / 3 : FLAGS_thread_num;
  m_thread_pool_ptr_ = std::shared_ptr<ThreadPool>(
      new ThreadPool(thread_num));
  return true;
}

bool Tree::init_pred_dist_vec(
    const double& p1, const double& p2, const double& p3) {
  std::vector<std::tuple<double, double, double>> pred_param_vec;
  auto init_param = std::make_tuple(p1, p2, p3);
  pred_param_vec.reserve(m_label_data_ptr_->size());
  for (unsigned int i = 0; i < m_label_data_ptr_->size(); ++i) {
    pred_param_vec.push_back(init_param);
  }
  m_pred_param_vec_ptr_ =
      std::make_shared<std::vector<std::tuple<double, double, double>>>(pred_param_vec);
  return true;
}

bool Tree::predict_one_tree(
    const boost::numeric::ublas::matrix_row<
    boost::numeric::ublas::compressed_matrix<double>>& record,
    const PBTree_Node& root, double* p1, double* p2, double* p3) {
  if (!(root.has_left_child() && root.left_child().has_level()) &&
      !(root.has_right_child() && root.right_child().has_level())) {
    *p1 = root.p1();
    *p2 = root.p2();
    VLOG(202) << "At level " << root.level() << " " << *p1 << " " << *p2;
    return true;
  }
  // We assume every node is binary tree
  CHECK(root.has_left_child() && root.has_right_child());
  double split_feature_value = root.split_feature_value();
  double split_feature_index = root.split_feature_index();
  if (Utility::check_double_le(
      record(split_feature_index), split_feature_value)) {
    VLOG(202) << "Go into left child " << root.level();
    predict_one_tree(record, root.left_child(), p1, p2, p3);
  } else {
    VLOG(202) << "Go into right child " << root.level();
    predict_one_tree(record, root.right_child(), p1, p2, p3);
  }
  return true;
}

bool Tree::predict(
    const boost::numeric::ublas::matrix_row<
    boost::numeric::ublas::compressed_matrix<double>>& record,
    double* p1, double* p2, double* p3) {
  if (m_pbtree_ptr_->tree_size() == 0) {
    return false;
  }
  const PBTree_Node& root = m_pbtree_ptr_->tree(0);

  if (!predict_one_tree(record, root, p1, p2, p3)) {
    LOG(ERROR) << "Predict instance failed";
    return false;
  };
  return true;
}

bool Tree::boost_predict_data_set(
  const boost::numeric::ublas::compressed_matrix<double>& matrix,
  std::vector<std::tuple<double, double, double>>* pred_param_vec,
  std::vector<std::tuple<double, double>>* pred_moment_vec,
  std::vector<std::pair<double, double>>* pred_interval_vec) {
  auto dist = DistributionManager::get_distribution(m_pbtree_ptr_->tree(0).distribution_type());
  pred_param_vec->reserve(matrix.size2());
  pred_moment_vec->reserve(matrix.size2());
  pred_interval_vec->reserve(matrix.size2());
  for (unsigned long i = 0; i < matrix.size2(); ++i) {  // Assumed column major
    double p1 = m_pbtree_ptr_->init_p1(), p2 = m_pbtree_ptr_->init_p2(), p3 = m_pbtree_ptr_->init_p3();
    for (int j = 0; j < m_pbtree_ptr_->tree_size(); ++j) {
      boost_update_one_instance(m_pbtree_ptr_->tree(j), i, &p1, &p2, &p3);
      VLOG(102) << "Round " << j << " param: (" << p1 << "," << p2 << ")";
    }
    double raw_param_p1 = 0, raw_param_p2 = 0;
    dist->transform_param(p1, p2, p3, &raw_param_p1, &raw_param_p2, nullptr);
    auto pred_param = std::make_tuple(raw_param_p1, raw_param_p2, 0.0);
    double first_moment = 0, second_moment = 0;
    dist->param_to_moment(pred_param, &first_moment, &second_moment);
    pred_param_vec->push_back(pred_param);
    pred_moment_vec->push_back(std::make_tuple(first_moment, second_moment));
    double lower_bound = 0, upper_bound = 0;
    dist->predict_interval(
        raw_param_p1, raw_param_p2, 0.0, FLAGS_lower_confidence_interval, FLAGS_upper_confidence_interval,
        &lower_bound, &upper_bound);
    pred_interval_vec->push_back(std::make_pair(lower_bound, upper_bound));
  }
  return true;
}

bool Tree::boost_update_one_instance(
    const PBTree_Node& new_tree_node,
    unsigned long record_index,
    double* p1, double* p2, double* p3) {
  if (!new_tree_node.has_left_child() && !new_tree_node.has_right_child()) {
    *p1 += new_tree_node.p1();
    *p2 += new_tree_node.p2();
    return true;
  }
  CHECK(new_tree_node.has_left_child() && new_tree_node.has_right_child());
  double split_feature_value = new_tree_node.split_feature_value();
  double split_feature_index = new_tree_node.split_feature_index();
  if (Utility::check_double_le(
      (*m_matrix_ptr_)(split_feature_index, record_index), split_feature_value)) {
    VLOG(202) << "Go into left child " << new_tree_node.level();
    boost_update_one_instance(new_tree_node.left_child(), record_index, p1, p2, p3);
  } else {
    VLOG(202) << "Go into right child " << new_tree_node.level();
    boost_update_one_instance(new_tree_node.right_child(), record_index, p1, p2, p3);
  }
  return true;
}

bool Tree::boost_update(const PBTree_Node& new_tree) {
  std::vector<std::tuple<double, double, double>> updated_param_vec;
  updated_param_vec.reserve(m_matrix_ptr_->size2());
  for (unsigned long i = 0; i < m_matrix_ptr_->size2(); ++i) {
    auto param = (*m_pred_param_vec_ptr_)[i];
    double p1 = std::get<0>(param);
    double p2 = std::get<1>(param);
    double p3 = std::get<2>(param);
    boost_update_one_instance(new_tree, i, &p1, &p2, &p3);
    updated_param_vec.push_back(std::make_tuple(p1, p2, p3));
  }
  m_pred_param_vec_ptr_ =
      std::make_shared<std::vector<
          std::tuple<double, double, double>>>(updated_param_vec);
  return true;
}

bool Tree::create_node(const std::vector<uint64_t>& record_index_vec,
    const uint32_t& level,
    PBTree_Node* node, const std::vector<uint64_t>* parents_feature_vec_ptr) {
  if (record_index_vec.size() < FLAGS_split_min_count ||
      level > FLAGS_tree_max_depth) {
    node->Clear();
    return false;
  }
  node->set_level(level);
  double current_record_ratio = record_index_vec.size() * 1.0 / m_matrix_ptr_->size2();
  LOG(INFO) << "Level " << level << " building start, record vec size = " << record_index_vec.size()
            << ", current record ratio = " << current_record_ratio;
  double current_loss = 0;
  if (FLAGS_boosting_mode) {
    m_distribution_ptr_->set_boost_node_param(
        *m_label_data_ptr_, record_index_vec, *m_pred_param_vec_ptr_, node);
    m_distribution_ptr_->calculate_boost_loss(*m_label_data_ptr_, record_index_vec, *m_pred_param_vec_ptr_, &current_loss, true);
  } else {
    m_distribution_ptr_->set_tree_node_param(*m_label_data_ptr_, record_index_vec, node);
    m_distribution_ptr_->calculate_loss(*m_label_data_ptr_, record_index_vec, &current_loss);
  }
  std::vector<uint64_t>* candidate_feature_vec_ptr = nullptr;
  std::vector<uint64_t> candidate_feature_vec;
  candidate_feature_vec.clear();
  if (current_record_ratio < FLAGS_record_min_ratio) {
    if (parents_feature_vec_ptr)
      check_valid_candidate(record_index_vec, *parents_feature_vec_ptr, &candidate_feature_vec);
    else 
      check_valid_candidate(record_index_vec, *m_valid_split_feature_vec_ptr_, &candidate_feature_vec);
    candidate_feature_vec_ptr = &candidate_feature_vec;
    LOG(INFO) << "Level " << level
              << ", parents candidate_feature_num "
              << std::to_string(parents_feature_vec_ptr ? parents_feature_vec_ptr->size() : 0)
              << ", v.s. all valid feature " << m_valid_split_feature_vec_ptr_->size()
              << ", after filter candidate_feature_vec size = " << candidate_feature_vec.size();
  }
  // if (Utility::check_double_le(current_loss, 0)) {
  //   LOG(INFO) << "Level " << level << " loss = " << current_loss << ", already converged";
  //   return true;
  // }
  uint64_t split_feature_index = 0;
  double split_point = 0;
  double split_loss = 0;
  bool split_ret = find_all_feature_split(
      record_index_vec, &split_feature_index, &split_point, &split_loss, candidate_feature_vec_ptr);
  if (!split_ret) {
    LOG(INFO) << "Level " << level << " find split feature failed!";
    return true;
  }
  CHECK(!std::isnan(current_loss));
  CHECK(!std::isnan(split_loss));
  LOG(INFO) << "Level " << level << ", record vec size = " << record_index_vec.size()
            << ", split_feature_index = " << split_feature_index
            << ", split_point = " << split_point
            << ", split loss = " << split_loss
            << ", current_loss = " << current_loss;
  double loss_param = record_index_vec.size() * 1.0 / (m_label_data_ptr_->size() * FLAGS_train_record_sample_ratio);
  if (!std::isinf(current_loss) && !std::isinf(split_loss) &&
      // !std::isnan(current_loss) && !std::isnan(split_loss) &&
//      split_loss / current_loss - 1 > -1 * FLAGS_split_gain_min_ratio) {
      (current_loss - split_loss) * loss_param < FLAGS_split_gain_min_value) {
    LOG(INFO) << "Level " << level << " split loss = " << split_loss
              << ", current_loss = " << current_loss
              << ", does not satisfy split_gain > " << FLAGS_split_gain_min_value;
    return true;
  }
  node->set_split_feature_index(split_feature_index);
  node->set_split_feature_value(split_point);
  // std::shared_ptr<PBTree_Node> left_child_ptr =
  //     std::shared_ptr<PBTree_Node>(new PBTree_Node());
  // std::shared_ptr<PBTree_Node> right_child_ptr =
  //     std::shared_ptr<PBTree_Node>(new PBTree_Node());
  std::vector<uint64_t> left_index_vec;
  left_index_vec.reserve(record_index_vec.size());
  std::vector<uint64_t> right_index_vec;
  right_index_vec.reserve(record_index_vec.size());
  for (auto iter = record_index_vec.begin(); iter != record_index_vec.end(); ++iter) {
    if (Utility::check_double_le(
        (*m_matrix_ptr_)(split_feature_index, *iter), split_point)) {
      left_index_vec.push_back(*iter);
    } else {
      right_index_vec.push_back(*iter);
    }
  }
  if (FLAGS_evaluate_loss_every_node) {
    double left_loss = 0, right_loss = 0;
    m_distribution_ptr_->calculate_boost_loss(*m_label_data_ptr_, left_index_vec, *m_pred_param_vec_ptr_, &left_loss, true);
    m_distribution_ptr_->calculate_boost_loss(*m_label_data_ptr_, right_index_vec, *m_pred_param_vec_ptr_, &right_loss, true);
    LOG(INFO) << "Level " << level << ", record vec size = " << record_index_vec.size()
              << ", split_feature_index = " << split_feature_index
              << ", split_point = " << split_point
              << ", split loss = " << split_loss
              << ", current_loss = " << current_loss
              << ", left_vec_size = " << left_index_vec.size()
              << ", left_loss = " << left_loss
              << ", right_vec_size = " << right_index_vec.size()
              << ", right_loss = " << right_loss;
  }
  if (left_index_vec.size() < FLAGS_split_min_count ||
      right_index_vec.size() < FLAGS_split_min_count) {
    LOG(INFO) << "Level " << level << " split failed "
              << ", left_index_vec.size() = " << left_index_vec.size()
              << ", right_index_vec.size() = " << right_index_vec.size()
              << ", less than min_split_count " << FLAGS_split_min_count;
    return true;
  }
  PBTree_Node* left_child = new PBTree_Node();
  PBTree_Node* right_child = new PBTree_Node();
  node->set_allocated_left_child(left_child);
  node->set_allocated_right_child(right_child);
  uint32_t next_level = level + 1;
  if (!create_node(left_index_vec, next_level, left_child, candidate_feature_vec_ptr)) {
    node->clear_left_child();
  }

  if (!create_node(right_index_vec, next_level, right_child, candidate_feature_vec_ptr)) {
    node->clear_right_child();
  }
  return true;
}

bool Tree::find_all_feature_split(
    const std::vector<uint64_t>& record_index_vec,
    uint64_t* split_feature_index, double* split_point,
    double* split_loss,
    const std::vector<uint64_t>* candidate_feature_vec) {
  if (record_index_vec.size() < FLAGS_split_min_count) {
    return false;
  }
  if (candidate_feature_vec && candidate_feature_vec->size() == 0) {return false;}
  std::vector<std::pair<uint64_t, std::pair<double, double>>> candidate_split_vec;
  // for (unsigned int i = 0; i < m_valid_histogram_vec_ptr_->size(); ++i) {
  //   uint64_t feature_index = (*m_valid_histogram_vec_ptr_)[i].first;
  //   (*m_candidate_split_vec_ptr_)[i] =
  //       std::make_pair(feature_index, std::make_pair(std::numeric_limits<double>::infinity(), DBL_MAX));
  // }
  // candidate_split_vec.reserve(m_valid_split_feature_vec_ptr_->size());
  // std::fill(candidate_split_vec.begin(), candidate_split_vec.end(), std::make_pair(0, std::make_pair(std::numeric_limits<double>::infinity(), DBL_MAX)));
  if (FLAGS_thread_num <= 1) {
    // for (unsigned long col_index = 0; col_index < m_matrix_ptr_->size1(); ++col_index) {
    for (auto iter = m_valid_split_feature_vec_ptr_->begin(); iter != m_valid_split_feature_vec_ptr_->end(); ++iter) {
      double tmp_split_point = 0, tmp_split_loss = DBL_MAX;
      uint64_t feature_index = *iter;
      // uint64_t iter_offset = iter - m_valid_split_feature_vec_ptr_->begin();
      if (find_one_feature_split(&record_index_vec, feature_index, &tmp_split_point, &tmp_split_loss))
        // (*m_candidate_split_vec_ptr_)[iter_offset] =
        //     std::make_pair(feature_index, std::make_pair(tmp_split_point, tmp_split_loss));
        candidate_split_vec.push_back(std::make_pair(feature_index, std::make_pair(tmp_split_point, tmp_split_loss)));
    }
  } else {
    // std::vector<std::pair<uint64_t, std::pair<double, double>>> tmp_candidate_split_vec;
    // for (auto iter = m_valid_histogram_vec_ptr_->begin(); iter != m_valid_histogram_vec_ptr_->end(); ++iter) {
    //   uint64_t feature_index = iter->first;
    //   candidate_split_vec.push_back(
    //     std::make_pair(feature_index, std::make_pair(std::numeric_limits<double>::infinity(), DBL_MAX)));
    // }
    uint64_t candidate_feature_num = 0;
    // std::unordered_set<uint64_t>* valid_feature_set_ptr;
    std::unordered_set<uint64_t> candidate_feature_set;
    if (candidate_feature_vec)
      candidate_feature_set = std::unordered_set<uint64_t>(candidate_feature_vec->begin(), candidate_feature_vec->end());
    for (unsigned int i = 0; i < m_valid_histogram_vec_ptr_->size(); ++i) {
      uint64_t feature_index = (*m_valid_histogram_vec_ptr_)[i].first;
      // uint64_t candidate_index = i;
      double current_record_ratio = record_index_vec.size() * 1.0 / (m_label_data_ptr_->size() * FLAGS_train_record_sample_ratio);
      double expected_feature_non_zero_ratio = current_record_ratio * FLAGS_feature_non_zero_convert_ratio;
      double feature_zero_ratio = (*m_valid_histogram_vec_ptr_)[i].second[0].second;
      VLOG(101) << "Feature_index = " << feature_index
              << ", current_record_ratio = " << current_record_ratio
              << ", expected_feature_non_zero_ratio = " << expected_feature_non_zero_ratio
              << ", feature_zero_ratio " << feature_zero_ratio;
      if (candidate_feature_vec && candidate_feature_set.find(feature_index) == candidate_feature_set.end())
        continue;
      if ((feature_zero_ratio > expected_feature_non_zero_ratio &&
          feature_zero_ratio < 1 - expected_feature_non_zero_ratio) || current_record_ratio < FLAGS_record_min_ratio) {
        candidate_split_vec.push_back(std::make_pair(feature_index, std::make_pair(std::numeric_limits<double>::infinity(), DBL_MAX)));
        ++candidate_feature_num;
      }
    }
    boost::asio::thread_pool pool(FLAGS_thread_num);
    for (unsigned int i = 0; i < candidate_split_vec.size(); ++i) {
      uint64_t feature_index = candidate_split_vec[i].first;
        // m_thread_pool_ptr_->do_job(std::bind(
        //     &Tree::find_one_feature_split, this, &record_index_vec, feature_index,
        //     &(candidate_split_vec[i].second.first),
        //     &(candidate_split_vec[i].second.second)));
      boost::asio::post(pool, std::bind(
            &Tree::find_one_feature_split, this, &record_index_vec, feature_index,
            &(candidate_split_vec[i].second.first),
            &(candidate_split_vec[i].second.second)));
    }
    pool.join();
    pool.stop();
    // for (unsigned int i = 0; i < m_valid_histogram_vec_ptr_->size(); ++i) {
    //   uint64_t feature_index = (*m_valid_histogram_vec_ptr_)[i].first;
    //   uint64_t candidate_index = i;
    //   double current_record_ratio = record_index_vec.size() * 1.0 / (m_label_data_ptr_->size() * FLAGS_train_record_sample_ratio);
    //   double expected_feature_non_zero_ratio = current_record_ratio * FLAGS_feature_non_zero_convert_ratio;
    //   double feature_zero_ratio = (*m_valid_histogram_vec_ptr_)[i].second[0].second;
    //   VLOG(101) << "Feature_index = " << feature_index
    //           << ", current_record_ratio = " << current_record_ratio
    //           << ", expected_feature_non_zero_ratio = " << expected_feature_non_zero_ratio
    //           << ", feature_zero_ratio " << feature_zero_ratio;
    //   if (candidate_feature_vec && candidate_feature_set.find(feature_index) == candidate_feature_set.end())
    //     continue;
    //   if (feature_zero_ratio > expected_feature_non_zero_ratio &&
    //       feature_zero_ratio < 1 - expected_feature_non_zero_ratio) {
    //     m_thread_pool_ptr_->do_job(std::bind(
    //         &Tree::find_one_feature_split, this, &record_index_vec, feature_index,
    //         &((*m_candidate_split_vec_ptr_)[candidate_index].second.first),
    //         &((*m_candidate_split_vec_ptr_)[candidate_index].second.second)));
    //     ++candidate_feature_num;
    //   }
    // }
    VLOG(1) << "In calculation candidate feature num = " << candidate_feature_num 
            << ", triming candidate_feature_set size = " << candidate_feature_set.size();
    // for (auto iter = m_valid_split_feature_vec_ptr_->begin(); iter != m_valid_split_feature_vec_ptr_->end(); ++iter) {
    //   uint64_t feature_index = *iter;
    //   uint64_t candidate_index = iter - m_valid_split_feature_vec_ptr_->begin();
    //   // Find split cost 100ms, push job to thread cost 0.1ms, which is very expensive for sparse feature
    //   m_thread_pool_ptr_->do_job(std::bind(
    //       &Tree::find_one_feature_split, this, &record_index_vec, feature_index,
    //       &(tmp_candidate_split_vec[candidate_index].second.first),
    //       &(tmp_candidate_split_vec[candidate_index].second.second)));
    // }
    // while (// !m_thread_pool_ptr_->is_jobs_queue_empty()
           //|| m_thread_pool_ptr_->is_working()
    //       !m_thread_pool_ptr_->is_finished()
    // ) {
      // usleep(1000);
      // std::this_thread::sleep_for(std::chrono::seconds(1));
      // std::cout << "Queue size = " << m_thread_pool_ptr_->get_jobs_queue_size() << std::endl;
      // std::cout << m_thread_pool_ptr_->get_worker_status() << std::endl;
      // Loop and waiting
    // }
    // candidate_split_vec.push_back(tmp_candidate_split_vec[best_split_feature_index]);
    // for (auto iter = m_valid_split_feature_vec_ptr_->begin(); iter != m_valid_split_feature_vec_ptr_->end(); ++iter) {
    //   uint64_t candidate_index = iter - m_valid_split_feature_vec_ptr_->begin();
    //   if (!std::isinf(tmp_candidate_split_vec[candidate_index].second.first))
    //     candidate_split_vec.push_back(tmp_candidate_split_vec[candidate_index]);
    // }
  }
  // if (candidate_split_vec.size() <= 0) {
  //   return false;
  // }
  
  // std::sort(candidate_split_vec.begin(), candidate_split_vec.end(),
  //     [](std::pair<uint64_t, std::pair<double, double>>& a,
  //     std::pair<uint64_t, std::pair<double, double>>& b){
  //     return (a.second.second < b.second.second);  // minimize the loss
  //     });
  // for (auto iter = candidate_split_vec.begin(); iter != candidate_split_vec.end(); ++iter) {
  //   VLOG(101) << "After sorting " << iter->first << "," << iter->second.first << "," << iter->second.second;
  // }
  uint64_t best_split_feature_index = 0;
  for (unsigned int i = 0; i < candidate_split_vec.size(); ++i) {
    if (candidate_split_vec[i].second.second < candidate_split_vec[best_split_feature_index].second.second &&
        !std::isinf(candidate_split_vec[i].second.first)) {
      best_split_feature_index = i;
    }
  }
  if (candidate_split_vec.empty() || std::isinf(candidate_split_vec[best_split_feature_index].second.first)) {
    return false;
  }
  *split_feature_index = candidate_split_vec[best_split_feature_index].first;
  *split_point = candidate_split_vec[best_split_feature_index].second.first;
  *split_loss = candidate_split_vec[best_split_feature_index].second.second;
  // for (unsigned int i = 0; i < m_candidate_split_vec_ptr_->size(); ++i) {
  //   if ((*m_candidate_split_vec_ptr_)[i].second.second < (*m_candidate_split_vec_ptr_)[best_split_feature_index].second.second &&
  //       !std::isinf((*m_candidate_split_vec_ptr_)[i].second.first)) {
  //     best_split_feature_index = i;
  //   }
  // }
  // if (std::isinf((*m_candidate_split_vec_ptr_)[best_split_feature_index].second.first)) {
  //   return false;
  // }
  // *split_feature_index = (*m_candidate_split_vec_ptr_)[best_split_feature_index].first;
  // *split_point = (*m_candidate_split_vec_ptr_)[best_split_feature_index].second.first;
  // *split_loss = (*m_candidate_split_vec_ptr_)[best_split_feature_index].second.second;
  return true;
}

bool Tree::check_split_histogram(const uint64_t& feature_index) {
  const std::vector<std::pair<double, float>>& histogram = (*m_histogram_vec_ptr_)[feature_index];
  if (histogram.empty()) {
    VLOG(202) << "Feature index " << feature_index << " empty histogram";
    return false;
  }
  if (histogram.size() == 1) {
    VLOG(201) << "Feature index histogram size is 1";
    return false;
  }
  if (histogram[0].second > 1 - FLAGS_split_min_ratio) {
    VLOG(101) << "Feature index " << feature_index << " zero ratio "
              << histogram[0].second << " > " << FLAGS_split_min_ratio;
    return false;
  }
  return true;
}

void Tree::do_intersection1(
    const std::vector<uint64_t>* record_index_vec_ptr,
    const std::vector<uint64_t>* pre_filtered_feature_vec_ptr,
    const uint32_t& begin_index,
    const uint32_t& end_index,
    std::vector<uint64_t>* result_vec_ptr) {
  std::vector<uint64_t> vec(record_index_vec_ptr->size() + m_max_non_zero_per_feature_);
  for (unsigned int i = begin_index; i < end_index; ++i) {
    auto tmp_non_zero_vec_iter = m_non_zero_value_map_ptr_->find((*pre_filtered_feature_vec_ptr)[i]);
    if (tmp_non_zero_vec_iter == m_non_zero_value_map_ptr_->end()) {
      std::cout << "Feature " << (*pre_filtered_feature_vec_ptr)[i] << " not found in feature map" << std::endl;
      for (unsigned j = begin_index; j < end_index; ++j) {
        std::cout << "Previous feature vec[" << j << "]" << (*pre_filtered_feature_vec_ptr)[j] << std::endl;
      }
      continue;
    }
    auto iter = std::set_intersection(
        tmp_non_zero_vec_iter->second.begin(),
        tmp_non_zero_vec_iter->second.end(),
        record_index_vec_ptr->begin(), record_index_vec_ptr->end(), vec.begin());
    (*result_vec_ptr)[i] = iter - vec.begin();
  }
  return;
}

void Tree::do_intersection(
    const std::vector<uint64_t>::const_iterator& iter_begin,
    const std::vector<uint64_t>::const_iterator& iter_end,
    const std::vector<uint64_t>* record_inded_vec_ptr,
    uint64_t* num) {
  std::vector<uint64_t> vec(record_inded_vec_ptr->size() + m_max_non_zero_per_feature_);
  for (auto iter_feature = iter_begin; iter_feature != iter_end; ++iter_feature) {
    auto tmp_non_zero_vec_iter = m_non_zero_value_map_ptr_->find(*iter_feature);
    // CHECK(tmp_non_zero_vec_iter != m_non_zero_value_map_ptr_->end());
    if (tmp_non_zero_vec_iter == m_non_zero_value_map_ptr_->end()) {
      LOG(ERROR) << "Feature " << *iter_feature << " not found in feature map";
      for (auto tmp_iter = iter_begin; tmp_iter != iter_end; ++tmp_iter) {
        VLOG(1) << "Previous feature vec[" << tmp_iter - iter_begin << "]" << *tmp_iter;
      }
      continue;
    }
    auto iter = std::set_intersection(
        tmp_non_zero_vec_iter->second.begin(),
        tmp_non_zero_vec_iter->second.end(),
        record_inded_vec_ptr->begin(), record_inded_vec_ptr->end(), vec.begin());
    *(num + (iter_feature - iter_begin)) = iter - vec.begin();
  }
  return;
}

bool Tree::check_valid_candidate(
    const std::vector<uint64_t> record_index_vec,
    const std::vector<uint64_t> pre_filter_feature_vec,
    std::vector<uint64_t>* candidate_feature_vec) {
  // Single thread implementation
  if (FLAGS_thread_num == 1 || !FLAGS_use_multi_thread_filter) {
    std::vector<uint64_t> vec(record_index_vec.size() + m_max_non_zero_per_feature_);
    for (auto iter_feature = pre_filter_feature_vec.begin();
        iter_feature != pre_filter_feature_vec.end(); ++iter_feature) {
      auto tmp_non_zero_vec_iter = m_non_zero_value_map_ptr_->find(*iter_feature);
      CHECK(tmp_non_zero_vec_iter != m_non_zero_value_map_ptr_->end());
      auto iter = std::set_intersection(
          tmp_non_zero_vec_iter->second.begin(),
          tmp_non_zero_vec_iter->second.end(),
          record_index_vec.begin(), record_index_vec.end(), vec.begin());
      if (iter - vec.begin() >= FLAGS_split_min_count) {
        candidate_feature_vec->push_back(*iter_feature);
      }
    }
    return true;
  }
  // Multi-thread implementation
  boost::asio::thread_pool pool(FLAGS_thread_num);
  std::vector<uint64_t> result_vec;
  result_vec.resize(pre_filter_feature_vec.size());
  uint32_t batch_size = FLAGS_filter_feature_batch_size;
  for (unsigned int i = 0; i < pre_filter_feature_vec.size(); i += batch_size) {
    unsigned int j = i + batch_size < pre_filter_feature_vec.size() ?
        i + batch_size : pre_filter_feature_vec.size();
    // m_thread_pool_ptr_->do_job(std::bind(
    //     &Tree::do_intersection1, this,
    //     &record_index_vec,
    //     &pre_filter_feature_vec,
    //     i, j, &result_vec
    // ));
    boost::asio::post(pool, std::bind(
        &Tree::do_intersection1, this,
        &record_index_vec,
        &pre_filter_feature_vec,
        i, j, &result_vec
    ));
  }
  pool.join();
  pool.stop();
  // for (auto iter_feature = pre_filter_feature_vec.begin();
  //     iter_feature < pre_filter_feature_vec.end(); iter_feature += batch_size) {
  //   auto current_turn_end = iter_feature + batch_size < pre_filter_feature_vec.end() ?
  //       iter_feature + batch_size : pre_filter_feature_vec.end();
  //   m_thread_pool_ptr_->do_job(
  //       std::bind(&Tree::do_intersection, this,
  //       iter_feature,
  //       current_turn_end,
  //       &record_index_vec,
  //       &(result_vec[iter_feature - pre_filter_feature_vec.begin()])
  //   ));
  // }
  //        while (!m_thread_pool_ptr_->is_finished()) {/*Just wait for finishing*/}
  for (unsigned long i = 0; i < pre_filter_feature_vec.size(); ++i) {
    if (result_vec[i] >= FLAGS_split_min_count) {
      candidate_feature_vec->push_back(pre_filter_feature_vec[i]);
    }
  }
  return true;
}

bool Tree::find_one_feature_split(
    const std::vector<uint64_t>* record_index_vec, const uint64_t& feature_index,
    double* split_point, double* split_loss) {
  const std::vector<std::pair<double, float>>& histogram = (*m_histogram_vec_ptr_)[feature_index];
  std::vector<std::pair<double, double>> candidate_split_vec;
  // if (histogram.empty()) {
  //   VLOG(202) << "Feature index " << feature_index << " empty histogram";
  //   return false;
  // }
  // if (histogram.size() == 1) {
  //   *split_point = 0;
  //   *split_loss = DBL_MAX;
  //   VLOG(201) << "Feature index histogram size is 1";
  //   return false;
  // }
  // if (histogram[0].second > 1 - FLAGS_split_min_ratio) {
  //   VLOG(101) << "Feature index " << feature_index << " zero ratio "
  //             << histogram[0].second << " > " << FLAGS_split_min_ratio;
  //   return true;
  // }
  // if (!check_split_histogram(feature_index)) {
  //   VLOG(102) << "Feature index " << feature_index << " can not build node.";
  //   return false;
  // }
  // double total_loss = 0;
  // m_distribution_ptr_->calculate_loss(*m_label_data_ptr_, row_index_vec, &total_loss);
  for (auto histogram_iter = histogram.begin();
      histogram_iter != histogram.end() - 1; ++histogram_iter) {

    uint64_t left_count = 0;
    for (auto row_iter = record_index_vec->begin(); row_iter != record_index_vec->end(); ++row_iter) {
      if (Utility::check_double_le((*m_matrix_ptr_)(feature_index, *row_iter), histogram_iter->first)) {
        ++left_count;
      }
    }
    uint64_t right_count = record_index_vec->size() - left_count;
    if (left_count < FLAGS_split_min_count ||
        right_count < FLAGS_split_min_count) {
      VLOG(101) << "Feature index " << feature_index
                << " not suitable for split_min_count " << FLAGS_split_min_count 
                << ", left count = " << left_count
                << ", right count = " << record_index_vec->size() - left_count;
      continue;
    }
    std::vector<uint64_t> left_index_vec;
    left_index_vec.reserve(left_count);
    std::vector<uint64_t> right_index_vec;
    right_index_vec.reserve(right_count);

    for (auto row_iter = record_index_vec->begin(); row_iter != record_index_vec->end(); ++row_iter) {
      if (Utility::check_double_le((*m_matrix_ptr_)(feature_index, *row_iter), histogram_iter->first)) {
        // left_label_vec.push_back((*m_label_data_ptr_)[*row_iter]);
        left_index_vec.push_back(*row_iter);
      } else {
        right_index_vec.push_back(*row_iter);
      }
    }
    double left_loss = 0;
    double right_loss = 0;
    if (FLAGS_boosting_mode) {
      m_distribution_ptr_->calculate_boost_loss(*m_label_data_ptr_, left_index_vec, *m_pred_param_vec_ptr_, &left_loss);
      m_distribution_ptr_->calculate_boost_loss(*m_label_data_ptr_, right_index_vec, *m_pred_param_vec_ptr_, &right_loss);
    } else {
      m_distribution_ptr_->calculate_loss(*m_label_data_ptr_, left_index_vec, &left_loss);
      m_distribution_ptr_->calculate_loss(*m_label_data_ptr_, right_index_vec, &right_loss);
    }

    double total_loss = left_index_vec.size() * left_loss + right_index_vec.size() * right_loss;
    total_loss /= record_index_vec->size();
    VLOG(101) << "Feature index: " << feature_index << " split point = " << histogram_iter->first
              << " left_index_vec.size() = " << left_index_vec.size()
              << " left_loss = " << left_loss
              << " right_index_vec.size() = " << right_index_vec.size()
              << " right_loss = " << right_loss;
    candidate_split_vec.push_back(std::make_pair(histogram_iter->first, total_loss));
  }
  if (candidate_split_vec.empty()) {
    VLOG(101) << "Feature index: " << feature_index << " candidate empty.";
    return false;
  }
  std::sort(candidate_split_vec.begin(), candidate_split_vec.end(),
      [](std::pair<double, double> a, std::pair<double, double> b)
      {return a.second < b.second;});
  *split_point = candidate_split_vec[0].first;
  *split_loss = candidate_split_vec[0].second;
  VLOG(102) << "col_index " << feature_index << " split point is " << *split_point
            << " split loss is " << *split_loss;
  return true;
}

bool Tree::build_histogram(
    const std::shared_ptr<boost::numeric::ublas::compressed_matrix<double>>& matrix_ptr,
    const uint64_t feature_index,
    const boost::numeric::ublas::compressed_matrix<double>::iterator1& feature_iter,
    std::vector<std::pair<double, float>>* histogram) {
  using namespace boost::numeric::ublas;
  VLOG(203) << "Begin building column";
  // matrix_column<compressed_matrix<double>> col = column(*matrix_ptr, feature_index);
  VLOG(203) << "Begin count non zero value";
  // auto col_iter = matrix_ptr->begin2();
  // col_iter = col_iter + feature_index;
  uint64_t non_zero_count = 0;
  for (auto iter = feature_iter.begin(); iter != feature_iter.end(); ++iter) {
    ++non_zero_count;
  }
  VLOG(203) << "Begin push non zero value";
  std::vector<double> vec;
  for (auto iter = feature_iter.begin(); iter != feature_iter.end(); ++iter) {
    vec.push_back(*iter);
  }
  VLOG(203) << "Feature index " << feature_index
             << " non zero count = " << non_zero_count;
  if (vec.size() * 1.0 / matrix_ptr->size2() < FLAGS_split_min_ratio) {
    histogram->push_back(std::make_pair(0.0, 1.0));
    VLOG(203) << "Non zero value too less";
    return true;
  }

  std::sort(vec.begin(), vec.end());
  std::vector<std::pair<double, float>> mid_histogram;
  uint32_t real_histogram_width = FLAGS_histogram_bin_count * vec.size() / matrix_ptr->size2();
  mid_histogram.push_back(std::make_pair(0.0, 1 - vec.size() * 1.0 / matrix_ptr->size2()));
  for (unsigned int i = 0; i < real_histogram_width; ++i) {
    int index = vec.size() * (i + 1) / real_histogram_width - 1;  // Zero based
    index = index < 0 ? 0 : index; 
    double value = vec[index];
    float portion = (index + 1) * 1.0 / matrix_ptr->size2();
    if (!Utility::check_double_le(value, 0.0)) {  // value > 0.0
      portion += mid_histogram[0].second;
    }
    mid_histogram.push_back(std::make_pair(value, portion));
  }
  for (auto iter = mid_histogram.begin(); iter < mid_histogram.end() - 1; ++iter) {
    auto next_iter = iter + 1;
    if (!Utility::check_double_equal(iter->first, next_iter->first)) {
      histogram->push_back(*iter);
    }
  }
  histogram->push_back(mid_histogram.back());
  return true;
}

bool Tree::build_tree() {
  // build histogram
  std::vector
      <std::vector<std::pair<double, float>>> histogram_vec;
  std::shared_ptr<std::vector
      <std::vector<std::pair<double, float>>>> histogram_vec_ptr = 
      std::make_shared<std::vector
      <std::vector<std::pair<double, float>>>>(histogram_vec);
  LOG(INFO) << "Begin building histogram";
  auto feature_iter = m_matrix_ptr_->begin1();
  for (unsigned long j = 0; j < m_matrix_ptr_->size1(); ++j) {
    std::vector<std::pair<double, float>> tmp_histogram;
    build_histogram(m_matrix_ptr_, j, feature_iter, &tmp_histogram);
    histogram_vec_ptr->push_back(tmp_histogram);
    ++feature_iter;
    if (j % 10000 == 0) {
      LOG(INFO) << "Built " << j << " histogram";
    }
  }
  for (auto iter1 = histogram_vec_ptr->begin(); iter1 != histogram_vec_ptr->end(); ++iter1) {
    for (auto iter2 = iter1->begin(); iter2 != iter1->end(); ++iter2) {
      VLOG(103) << iter1 - histogram_vec_ptr->begin() << "," << iter2 - iter1->begin() << ":"
                << iter2->first << " " << iter2->second;
    }
  }
  m_histogram_vec_ptr_ = histogram_vec_ptr;
  LOG(INFO) << "Finished building histogram";

  std::vector<uint64_t> valid_split_feature_vec;
  std::vector<std::pair<uint64_t, std::vector<std::pair<double, float>>>> valid_histogram_vec;
  std::map<uint64_t, std::vector<uint64_t>> non_zero_value_map;
  feature_iter = m_matrix_ptr_->begin1();
  uint64_t valid_total_nnz = 0;
  for (unsigned long feature_index = 0; feature_index < m_matrix_ptr_->size1(); ++feature_index) {
    if (check_split_histogram(feature_index)) {
      valid_split_feature_vec.push_back(feature_index);
      valid_histogram_vec.push_back(std::make_pair(feature_index, (*m_histogram_vec_ptr_)[feature_index]));
      std::vector<uint64_t> tmp_vec;
      for (auto record_iter = feature_iter.begin(); record_iter != feature_iter.end(); ++record_iter) {
        tmp_vec.push_back(record_iter.index2());
        VLOG(101) << "Record iter " << record_iter.index1() << " " << record_iter.index2();
      }
      // std::sort(tmp_vec.begin(), tmp_vec.end());
      non_zero_value_map[feature_index] = tmp_vec;
      if (tmp_vec.size() > m_max_non_zero_per_feature_) {
        m_max_non_zero_per_feature_ = tmp_vec.size();
      }
      valid_total_nnz += tmp_vec.size();
    }
    ++feature_iter;
  }
  LOG(INFO) << "Max feature-wise nnz = " << m_max_non_zero_per_feature_
            << " valid total_nnz = " << valid_total_nnz;
  m_non_zero_value_map_ptr_ =
      std::make_shared<std::map<uint64_t, std::vector<uint64_t>>>(non_zero_value_map);
  // std::sort(valid_histogram_vec.begin(), valid_histogram_vec.end(),
  //     [](const std::pair<uint64_t, std::vector<std::pair<double, float>>>& a,
  //     const std::pair<uint64_t, std::vector<std::pair<double, float>>>& b){
  //       return a.second[0].second < b.second[0].second;
  //     });
  m_valid_histogram_vec_ptr_ =
      std::make_shared<std::vector<std::pair<uint64_t,
          std::vector<std::pair<double, float>>>>>(valid_histogram_vec);
  
  for (auto iter1 = histogram_vec_ptr->begin();
      (iter1 != histogram_vec_ptr->begin() + 500) && iter1 != histogram_vec_ptr->end(); ++iter1) {
    for (auto iter2 = iter1->begin(); iter2 != iter1->end(); ++iter2) {
      LOG(INFO) << iter1 - histogram_vec_ptr->begin() << "," << iter2 - iter1->begin() << ":"
                << iter2->first << " " << iter2->second;
    }
  }

  for (auto iter1 = histogram_vec_ptr->begin(); iter1 != histogram_vec_ptr->end(); ++iter1) {
    if (check_split_histogram(iter1 - histogram_vec_ptr->begin())) {
      for (auto iter2 = iter1->begin(); iter2 != iter1->end(); ++iter2) {
        LOG(INFO) << iter1 - histogram_vec_ptr->begin() << "," << iter2 - iter1->begin() << ":"
                  << iter2->first << " " << iter2->second;
      }
    }
  }

  for (auto iter1 = m_valid_histogram_vec_ptr_->begin(); iter1 != m_valid_histogram_vec_ptr_->end(); ++iter1) {
    if (check_split_histogram(iter1 - m_valid_histogram_vec_ptr_->begin())) {
      for (auto iter2 = iter1->second.begin(); iter2 != iter1->second.end(); ++iter2) {
        LOG(INFO) << "Feature index " << iter1->first << ","
                  << iter1 - m_valid_histogram_vec_ptr_->begin() << ","
                  << iter2 - iter1->second.begin() << ":"
                  << iter2->first << " " << iter2->second;
      }
    }
  }

  m_valid_split_feature_vec_ptr_ = std::make_shared<std::vector<uint64_t>>(valid_split_feature_vec);
  LOG(INFO) << "Valid feature index size is " << m_valid_split_feature_vec_ptr_->size();
  // std::shared_ptr<PBTree_Node> root =
  //     std::shared_ptr<PBTree_Node>(new PBTree_Node());
  // std::vector<std::pair<uint64_t, std::pair<double, double>>> candidate_split_vec;
  // candidate_split_vec.resize(m_valid_histogram_vec_ptr_->size());
  // for (unsigned int i = 0; i < m_valid_histogram_vec_ptr_->size(); ++i) {
  //   uint64_t feature_index = (*m_valid_histogram_vec_ptr_)[i].first;
  //   candidate_split_vec[i] =
  //     std::make_pair(feature_index, std::make_pair(std::numeric_limits<double>::infinity(), DBL_MAX));
  // }
  // m_candidate_split_vec_ptr_ =
  //     std::make_shared<std::vector<std::pair<uint64_t, std::pair<double, double>>>>(candidate_split_vec);

  std::vector<uint64_t> record_index_vec;
  record_index_vec.reserve(m_matrix_ptr_->size2());
  for (unsigned long i = 0; i < m_matrix_ptr_->size2(); ++i) {
    record_index_vec.push_back(i);
  }
  ModelManager model_manager;
  // if (!FLAGS_burned_model_path.empty()) {
  //   model_manager.load_tree_model(FLAGS_burned_model_path, &m_pbtree_ptr_);
  // } else {
  //   m_pbtree_ptr_ = std::make_shared<pbtree::PBTree>(new pbtree::PBTree());
  // }
  double learning_rate1 = FLAGS_learning_rate1;
  double learning_rate2 = FLAGS_learning_rate2;
  double init_p1 = 0, init_p2 = 0, init_p3 = 0;
  m_distribution_ptr_->init_param(&init_p1, &init_p2, &init_p3);
  m_pbtree_ptr_->set_init_p1(init_p1);
  m_pbtree_ptr_->set_init_p2(init_p2);
  m_pbtree_ptr_->set_init_p3(init_p3);
  init_pred_dist_vec(init_p1, init_p2, init_p3);
  for (unsigned int i = 0; i < FLAGS_training_round; ++i) {
    m_distribution_ptr_->get_learning_rate(i, learning_rate1, learning_rate2, 0,
        &FLAGS_learning_rate1, &FLAGS_learning_rate2, nullptr);
    PBTree_Node* root = m_pbtree_ptr_->add_tree();
    std::vector<uint64_t> train_index_vec;
    if (Utility::check_double_equal(FLAGS_train_record_sample_ratio, 1.0)) {
      train_index_vec = record_index_vec;
    } else {
      srand((unsigned int)(time(nullptr)));
      for (auto iter = record_index_vec.begin(); iter != record_index_vec.end(); ++iter) {
        if (rand() / double(RAND_MAX) < FLAGS_train_record_sample_ratio) {
          train_index_vec.push_back(*iter);
        }
      }
    }
    LOG(INFO) << "Begin training round " << i << " with " << train_index_vec.size() << " instances"
              << ", learning_rate1 = " << FLAGS_learning_rate1
              << ", learning_rate2 = " << FLAGS_learning_rate2;
    create_node(train_index_vec, 0/*node level*/, root);
    if ((i + 1) % FLAGS_save_temp_model_round == 0) {
      std::string temp_output_model_path = FLAGS_output_model_path + ".temp_round_" + std::to_string(i);
      model_manager.save_tree_model(*m_pbtree_ptr_, temp_output_model_path);
    }
    if (FLAGS_boosting_mode) {
      boost_update(*root);
      double loss = 0;
      m_distribution_ptr_->calculate_boost_loss(
          *m_label_data_ptr_, record_index_vec, *m_pred_param_vec_ptr_, &loss, true);
      double rmsle = 0;
      m_distribution_ptr_->evaluate_rmsle(
          *m_label_data_ptr_, record_index_vec, *m_pred_param_vec_ptr_, &rmsle);
      LOG(INFO) << "Finished training the " << i << "-th round, loss = " << loss << ", rmsle = " << rmsle;
    }
  }
  return true;
}

}  // namespace pbtree
