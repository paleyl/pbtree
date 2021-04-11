#include "tree.h"

DEFINE_uint32(histogram_bin_count, 100, "Number of bins in histogram");
DEFINE_uint32(split_min_count, 5, "Minimum record count for split");
DEFINE_double(split_min_ratio, 0.02, "Minimum record count for split");
DEFINE_uint32(tree_max_depth, 10, "Minimum record count for split");

namespace pbtree {

bool Tree::predict_one_tree(
    const boost::numeric::ublas::matrix_row<
    boost::numeric::ublas::compressed_matrix<double>>& record,
    const PBTree_Node& root, double* p1, double* p2, double* p3) {
  if (!(root.has_left_child() && root.left_child().has_level()) ||
      !(root.has_right_child() && root.right_child().has_level())) {
    *p1 = root.p1();
    *p2 = root.p2();
    VLOG(202) << "At level " << root.level() << " " << *p1 << " " << *p2;
    return true;
  }
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

bool Tree::create_node(const std::vector<uint64_t>& row_index_vec,
    const uint32_t& level,
    PBTree_Node* node) {
  if (row_index_vec.size() < FLAGS_split_min_count ||
      level > FLAGS_tree_max_depth) {
    node->Clear();
    return true;
  };
  LOG(INFO) << "Building node on level " << level;
  uint64_t split_feature_index = 0;
  double split_point = 0;
  double split_loss = 0;
  find_all_feature_split(
      row_index_vec, &split_feature_index, &split_point, &split_loss);
  node->set_level(level);
  node->set_split_feature_index(split_feature_index);
  node->set_split_feature_value(split_point);
  LOG(INFO) << "Level " << level << " split_feature_index = " << split_feature_index
            << " split_point = " << split_point;
  m_distribution_ptr_->set_tree_node_param(*m_label_data_ptr_, row_index_vec, node);
  // std::shared_ptr<PBTree_Node> left_child_ptr =
  //     std::shared_ptr<PBTree_Node>(new PBTree_Node());
  // std::shared_ptr<PBTree_Node> right_child_ptr =
  //     std::shared_ptr<PBTree_Node>(new PBTree_Node());
  PBTree_Node* left_child = new PBTree_Node();
  PBTree_Node* right_child = new PBTree_Node();
  node->set_allocated_left_child(left_child);
  node->set_allocated_right_child(right_child);
  std::vector<uint64_t> left_index_vec;
  std::vector<uint64_t> right_index_vec;
  for (unsigned int row_index = 0; row_index < m_matrix_ptr_->size2(); ++row_index) {
    if (Utility::check_double_le(
        (*m_matrix_ptr_)(split_feature_index, row_index), split_point)) {
      left_index_vec.push_back(row_index);
    } else {
      right_index_vec.push_back(row_index);
    }
  }
  uint32_t next_level = level + 1;
  create_node(left_index_vec, next_level, left_child);
  create_node(right_index_vec, next_level, right_child);
  // if ()
  return true;
}

bool Tree::find_all_feature_split(
    const std::vector<uint64_t>& row_index_vec,
    uint64_t* split_feature_index, double* split_point,
    double* split_loss) {
  std::vector<std::pair<uint64_t, std::pair<double, double>>> candidate_split_vec;
  for (unsigned long col_index = 0; col_index < m_matrix_ptr_->size1(); ++col_index) {
    double tmp_split_point = 0, tmp_split_loss = 0;
    if (find_one_feature_split(row_index_vec, col_index, &tmp_split_point, &tmp_split_loss))
      candidate_split_vec.push_back(
          std::make_pair(col_index, std::make_pair(tmp_split_point, tmp_split_loss)));
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
    const std::vector<uint64_t>& row_index_vec, const uint64_t& col_index,
    double* split_point, double* split_loss) {
  const std::vector<std::pair<double, float>>& histogram = (*m_histogram_vec_ptr_)[col_index];
  std::vector<std::pair<double, double>> candidate_split_vec;
  if (histogram.empty()) {
    VLOG(102) << "Col index " << col_index << " empty histogram";
    return false;
  }
  if (histogram.size() == 1) {
    *split_point = 0;
    *split_loss = DBL_MAX;
    return false;
  }
  if (histogram[0].second > 1 - FLAGS_split_min_ratio) {
    return true;
  }
  // double total_loss = 0;
  // m_distribution_ptr_->calculate_loss(*m_label_data_ptr_, row_index_vec, &total_loss);
  for (auto histogram_iter = histogram.begin();
      histogram_iter != histogram.end() - 1; ++histogram_iter) {

    std::vector<uint64_t> left_index_vec;
    std::vector<uint64_t> right_index_vec;
    for (auto row_iter = row_index_vec.begin(); row_iter != row_index_vec.end(); ++row_iter) {
      if (Utility::check_double_le((*m_matrix_ptr_)(col_index, *row_iter), histogram_iter->first)) {
        // left_label_vec.push_back((*m_label_data_ptr_)[*row_iter]);
        left_index_vec.push_back(*row_iter);
      } else {
        right_index_vec.push_back(*row_iter);
      }
    }
    if (left_index_vec.size() < FLAGS_split_min_count ||
        right_index_vec.size() < FLAGS_split_min_count) {
      return false;
    }
    double left_loss = 0;
    m_distribution_ptr_->calculate_loss(*m_label_data_ptr_, left_index_vec, &left_loss);
    double right_loss = 0;
    m_distribution_ptr_->calculate_loss(*m_label_data_ptr_, right_index_vec, &right_loss);

    double total_loss = left_index_vec.size() * left_loss + right_index_vec.size() * right_loss;
    VLOG(101) << "Col index: " << col_index << " split point = " << histogram_iter->first
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
  VLOG(102) << "col_index " << col_index << " split point is " << *split_point
            << " split loss is " << *split_loss;
  return true;
}

bool Tree::build_histogram(
    const std::shared_ptr<boost::numeric::ublas::compressed_matrix<double>>& matrix_ptr,
    const uint64_t feature_index,
    const boost::numeric::ublas::compressed_matrix<double>::iterator1& col_iter,
    std::vector<std::pair<double, float>>* histogram) {
  using namespace boost::numeric::ublas;
  VLOG(103) << "Begin building column";
  // matrix_column<compressed_matrix<double>> col = column(*matrix_ptr, feature_index);
  VLOG(103) << "Begin count non zero value";
  // auto col_iter = matrix_ptr->begin2();
  // col_iter = col_iter + feature_index;
  uint64_t non_zero_count = 0;
  for (auto iter = col_iter.begin(); iter != col_iter.end(); ++iter) {
    ++non_zero_count;
  }
  VLOG(103) << "Begin push non zero value";
  std::vector<double> vec;
  for (auto iter = col_iter.begin(); iter != col_iter.end(); ++iter) {
    vec.push_back(*iter);
  }
  VLOG(103) << "Feature index " << feature_index
             << " non zero count = " << non_zero_count;
  if (vec.size() * 1.0 / matrix_ptr->size2() < FLAGS_split_min_ratio) {
    histogram->push_back(std::make_pair(0.0, 1.0));
    VLOG(103) << "Non zero value too less";
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
  PBTree_Node* root = m_pbtree_ptr_->add_tree();
  std::vector<uint64_t> row_index_vec;
  for (unsigned long i = 0; i < m_matrix_ptr_->size2(); ++i) {
    row_index_vec.push_back(i);
  }
  create_node(row_index_vec, 0, root);

  return true;
}

}  // namespace pbtree