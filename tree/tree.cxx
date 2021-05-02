#include "tree.h"

DEFINE_uint32(histogram_bin_count, 100, "Number of bins in histogram");
DEFINE_uint32(split_min_count, 5, "Minimum record count for split");
DEFINE_double(split_min_ratio, 0.02, "Minimum record count for split");
DEFINE_uint32(tree_max_depth, 10, "Minimum record count for split");
DEFINE_bool(boosting_mode, true, "");
DEFINE_uint32(training_round, 10, "");
DEFINE_double(split_gain_min_ratio, 0.01, "");

// TODO(paleylv): develop pthread strategy
// TODO(paleylv): add min gain ratio threshold
namespace pbtree {

bool Tree::init_pred_dist_vec() {
  std::vector<std::tuple<double, double, double>> pred_param_vec;
  double p1 = 0, p2 = 0, p3 = 0;
  m_distribution_ptr_->init_param(&p1, &p2, &p3);
  auto init_param = std::make_tuple(p1, p2, p3);
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
  std::vector<std::tuple<double, double>>* pred_moment_vec) {
  auto dist = DistributionManager::get_distribution(m_pbtree_ptr_->tree(0).distribution_type());
  for (unsigned long i = 0; i < matrix.size2(); ++i) {  // Assumed column major
    double p1 = 0, p2 = 0, p3 = 0;
    dist->init_param(&p1, &p2, &p3);
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
    PBTree_Node* node) {
  if (record_index_vec.size() < FLAGS_split_min_count ||
      level > FLAGS_tree_max_depth) {
    node->Clear();
    return false;
  }
  node->set_level(level);
  LOG(INFO) << "Building node on level " << level;
  LOG(INFO) << "Record vec size = " << record_index_vec.size();
  double current_loss = 0;
  if (FLAGS_boosting_mode) {
    m_distribution_ptr_->set_boost_node_param(
        *m_label_data_ptr_, record_index_vec, *m_pred_param_vec_ptr_, node);
    m_distribution_ptr_->calculate_boost_loss(*m_label_data_ptr_, record_index_vec, *m_pred_param_vec_ptr_, &current_loss, true);
  } else {
    m_distribution_ptr_->set_tree_node_param(*m_label_data_ptr_, record_index_vec, node);
    m_distribution_ptr_->calculate_loss(*m_label_data_ptr_, record_index_vec, &current_loss);
  }
  if (Utility::check_double_le(current_loss, 0)) {
    LOG(INFO) << "Level " << level << " loss = " << current_loss << ", already converged";
    return true;
  }
  uint64_t split_feature_index = 0;
  double split_point = 0;
  double split_loss = 0;
  bool split_ret = find_all_feature_split(
      record_index_vec, &split_feature_index, &split_point, &split_loss);
  if (!split_ret) {
    LOG(INFO) << "Level " << level << " find split feature failed!";
    return true;
  }
  CHECK(!std::isnan(current_loss));
  CHECK(!std::isnan(split_loss));
  LOG(INFO) << "Level " << level << " split loss = " << split_loss
            << ", current_loss = " << current_loss;
  if (!std::isinf(current_loss) && !std::isinf(split_loss) &&
      // !std::isnan(current_loss) && !std::isnan(split_loss) &&
      split_loss / current_loss - 1 > -1 * FLAGS_split_gain_min_ratio) {
    LOG(INFO) << "Level " << level << " split loss = " << split_loss
              << ", current_loss = " << current_loss
              << ", does not satisfy split_gain > " << FLAGS_split_gain_min_ratio;
    return true;
  }
  node->set_split_feature_index(split_feature_index);
  node->set_split_feature_value(split_point);
  LOG(INFO) << "Level " << level << " split_feature_index = " << split_feature_index
            << " split_point = " << split_point;
  // std::shared_ptr<PBTree_Node> left_child_ptr =
  //     std::shared_ptr<PBTree_Node>(new PBTree_Node());
  // std::shared_ptr<PBTree_Node> right_child_ptr =
  //     std::shared_ptr<PBTree_Node>(new PBTree_Node());
  std::vector<uint64_t> left_index_vec;
  std::vector<uint64_t> right_index_vec;
  for (auto iter = record_index_vec.begin(); iter != record_index_vec.end(); ++iter) {
    if (Utility::check_double_le(
        (*m_matrix_ptr_)(split_feature_index, *iter), split_point)) {
      left_index_vec.push_back(*iter);
    } else {
      right_index_vec.push_back(*iter);
    }
  }
  if (left_index_vec.size() < FLAGS_split_min_count ||
      right_index_vec.size() < FLAGS_split_min_count) {
    LOG(INFO) << "Level " << level << " split failed "
              << ", left_index_vec.size() = " << left_index_vec.size()
              << ", right_index_vec.size() = " << right_index_vec.size()
              << ", less than min_split_count" << FLAGS_split_min_count;
    return true;
  }
  PBTree_Node* left_child = new PBTree_Node();
  PBTree_Node* right_child = new PBTree_Node();
  node->set_allocated_left_child(left_child);
  node->set_allocated_right_child(right_child);
  uint32_t next_level = level + 1;
  if (!create_node(left_index_vec, next_level, left_child)) {
    node->clear_left_child();
  }

  if (!create_node(right_index_vec, next_level, right_child)) {
    node->clear_right_child();
  }
  return true;
}

bool Tree::find_all_feature_split(
    const std::vector<uint64_t>& record_index_vec,
    uint64_t* split_feature_index, double* split_point,
    double* split_loss) {
  if (record_index_vec.size() < FLAGS_split_min_count) {
    return false;
  }
  std::vector<std::pair<uint64_t, std::pair<double, double>>> candidate_split_vec;
  for (unsigned long col_index = 0; col_index < m_matrix_ptr_->size1(); ++col_index) {
    double tmp_split_point = 0, tmp_split_loss = 0;
    if (find_one_feature_split(record_index_vec, col_index, &tmp_split_point, &tmp_split_loss))
      candidate_split_vec.push_back(
          std::make_pair(col_index, std::make_pair(tmp_split_point, tmp_split_loss)));
  }
  if (candidate_split_vec.size() <= 0) {
    return false;
  }
  std::sort(candidate_split_vec.begin(), candidate_split_vec.end(),
      [](std::pair<uint64_t, std::pair<double, double>>& a,
      std::pair<uint64_t, std::pair<double, double>>& b){
      return a.second.second < b.second.second;  // minimize the loss
      });
  *split_feature_index = candidate_split_vec[0].first;
  *split_point = candidate_split_vec[0].second.first;
  *split_loss = candidate_split_vec[0].second.second;
  return true;
}

bool Tree::find_one_feature_split(
    const std::vector<uint64_t>& record_index_vec, const uint64_t& feature_index,
    double* split_point, double* split_loss) {
  const std::vector<std::pair<double, float>>& histogram = (*m_histogram_vec_ptr_)[feature_index];
  std::vector<std::pair<double, double>> candidate_split_vec;
  if (histogram.empty()) {
    VLOG(202) << "Feature index " << feature_index << " empty histogram";
    return false;
  }
  if (histogram.size() == 1) {
    *split_point = 0;
    *split_loss = DBL_MAX;
    VLOG(201) << "Feature index histogram size is 1";
    return false;
  }
  if (histogram[0].second > 1 - FLAGS_split_min_ratio) {
    VLOG(101) << "Feature index " << feature_index << " zero ratio "
              << histogram[0].second << " > " << FLAGS_split_min_ratio;
    return true;
  }
  // double total_loss = 0;
  // m_distribution_ptr_->calculate_loss(*m_label_data_ptr_, row_index_vec, &total_loss);
  for (auto histogram_iter = histogram.begin();
      histogram_iter != histogram.end() - 1; ++histogram_iter) {

    std::vector<uint64_t> left_index_vec;
    std::vector<uint64_t> right_index_vec;
    for (auto row_iter = record_index_vec.begin(); row_iter != record_index_vec.end(); ++row_iter) {
      if (Utility::check_double_le((*m_matrix_ptr_)(feature_index, *row_iter), histogram_iter->first)) {
        // left_label_vec.push_back((*m_label_data_ptr_)[*row_iter]);
        left_index_vec.push_back(*row_iter);
      } else {
        right_index_vec.push_back(*row_iter);
      }
    }
    if (left_index_vec.size() < FLAGS_split_min_count ||
        right_index_vec.size() < FLAGS_split_min_count) {
      LOG(INFO) << "Feature index " << feature_index
                << " not suitable for split_min_count " << FLAGS_split_min_count 
                << ", left count = " << left_index_vec.size()
                << ", right count = " << right_index_vec.size();
      return false;
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
    total_loss /= record_index_vec.size();
    VLOG(101) << "Feature index: " << feature_index << " split point = " << histogram_iter->first
              << " left_index_vec.size() = " << left_index_vec.size()
              << " left_loss = " << left_loss
              << " right_index_vec.size() = " << right_index_vec.size()
              << " right_loss = " << right_loss;
    candidate_split_vec.push_back(std::make_pair(histogram_iter->first, total_loss));
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
  auto col_iter = m_matrix_ptr_->begin1();
  for (unsigned long j = 0; j < m_matrix_ptr_->size1(); ++j) {
    std::vector<std::pair<double, float>> tmp_histogram;
    build_histogram(m_matrix_ptr_, j, col_iter, &tmp_histogram);
    histogram_vec_ptr->push_back(tmp_histogram);
    ++col_iter;
    if (j % 10000 == 0) {
      LOG(INFO) << "Built " << j << " histogram";
    }
  }
  for (auto iter1 = histogram_vec_ptr->begin(); iter1 != histogram_vec_ptr->end(); ++iter1) {
    for (auto iter2 = iter1->begin(); iter2 != iter1->end(); ++iter2) {
      VLOG(202) << iter1 - histogram_vec_ptr->begin() << "," << iter2 - iter1->begin() << ":"
                << iter2->first << " " << iter2->second;
    }
  }
  m_histogram_vec_ptr_ = histogram_vec_ptr;
  LOG(INFO) << "Finished building histogram";
  // std::shared_ptr<PBTree_Node> root =
  //     std::shared_ptr<PBTree_Node>(new PBTree_Node());

  std::vector<uint64_t> feature_index_vec;
  for (unsigned long i = 0; i < m_matrix_ptr_->size2(); ++i) {
    feature_index_vec.push_back(i);
  }

  for (unsigned int i = 0; i < FLAGS_training_round; ++i) {
    PBTree_Node* root = m_pbtree_ptr_->add_tree();
    create_node(feature_index_vec, 0/*node level*/, root);
    if (FLAGS_boosting_mode) {
      boost_update(*root);
      double loss = 0;
      m_distribution_ptr_->calculate_boost_loss(
          *m_label_data_ptr_, feature_index_vec, *m_pred_param_vec_ptr_, &loss, true);
      LOG(INFO) << "Finished training the " << i << "-th round, loss = " << loss;
    }
  }
  return true;
}

}  // namespace pbtree