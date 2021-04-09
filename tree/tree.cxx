#include "tree.h"

DEFINE_uint32(histogram_bin_count, 100, "Number of bins in histogram");
DEFINE_uint32(split_min_count, 5, "Minimum record count for split");
DEFINE_uint32(tree_max_depth, 10, "Minimum record count for split");

namespace pbtree {

bool Tree::create_node(const std::vector<uint64_t>& row_index_vec,
    const uint32_t& level,
    std::shared_ptr<PBTree_Node>* node) {
  if (row_index_vec.size() < FLAGS_split_min_count ||
      level > FLAGS_tree_max_depth) {
    *node = nullptr;
    return true;
  };
  uint64_t split_feature_index = 0;
  double split_point = 0;
  double split_loss = 0;
  find_all_feature_split(
      row_index_vec, &split_feature_index, &split_point, &split_loss);
  (*node)->set_level(level);
  (*node)->set_split_feature_index(split_feature_index);
  (*node)->set_split_feature_value(split_point);
  std::shared_ptr<PBTree_Node> left_child_ptr =
      std::shared_ptr<PBTree_Node>(new PBTree_Node());
  std::shared_ptr<PBTree_Node> right_child_ptr =
      std::shared_ptr<PBTree_Node>(new PBTree_Node());
  (*node)->set_allocated_left_child(left_child_ptr.get());
  (*node)->set_allocated_right_child(right_child_ptr.get());
  std::vector<uint64_t> left_index_vec;
  std::vector<uint64_t> right_index_vec;
  for (unsigned int row_index = 0; row_index < m_matrix_ptr_->size1(); ++row_index) {
    if (Utility::check_double_le(
        (*m_matrix_ptr_)(row_index, split_feature_index), split_point)) {
      left_index_vec.push_back(row_index);
    } else {
      right_index_vec.push_back(row_index);
    }
  }
  uint32_t next_level = level + 1;
  create_node(left_index_vec, next_level, &left_child_ptr);
  create_node(right_index_vec, next_level, &right_child_ptr);
  return true;
}

bool Tree::find_all_feature_split(
    const std::vector<uint64_t>& row_index_vec,
    uint64_t* split_feature_index, double* split_point,
    double* split_loss) {
  std::vector<std::pair<uint64_t, std::pair<double, double>>> candidate_split_vec;
  for (unsigned long col_index = 0; col_index < m_matrix_ptr_->size2(); ++col_index) {
    double tmp_split_point = 0, tmp_split_loss = 0;
    find_one_feature_split(row_index_vec, col_index, &tmp_split_point, &tmp_split_loss);
    candidate_split_vec.push_back(std::make_pair(col_index, std::make_pair(tmp_split_point, tmp_split_loss)));
  }
  std::sort(candidate_split_vec.begin(), candidate_split_vec.end(),
      [](std::pair<uint64_t, std::pair<double, double>> a, std::pair<uint64_t, std::pair<double, double>> b){
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
  // double total_loss = 0;
  // m_distribution_ptr_->calculate_loss(*m_label_data_ptr_, row_index_vec, &total_loss);
  for (auto histogram_iter = histogram.begin();
      histogram_iter != histogram.end() - 1; ++histogram_iter) {

    std::vector<uint64_t> left_index_vec;
    std::vector<uint64_t> right_index_vec;
    for (auto row_iter = row_index_vec.begin(); row_iter != row_index_vec.end(); ++row_iter) {
      if (Utility::check_double_le((*m_label_data_ptr_)[*row_iter], histogram_iter->first)) {
        // left_label_vec.push_back((*m_label_data_ptr_)[*row_iter]);
        left_index_vec.push_back(*row_iter);
      } else {
        right_index_vec.push_back(*row_iter);
      }
    }
    double left_loss = 0;
    m_distribution_ptr_->calculate_loss(*m_label_data_ptr_, left_index_vec, &left_loss);
    double right_loss = 0;
    m_distribution_ptr_->calculate_loss(*m_label_data_ptr_, right_index_vec, &right_loss);

    double total_loss = left_index_vec.size() * left_loss + right_index_vec.size() * right_loss;
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
    const std::shared_ptr<boost::numeric::ublas::mapped_matrix<double>>& matrix_ptr,
    const uint64_t feature_index,
    std::vector<std::pair<double, float>>* histogram) {
  using namespace boost::numeric::ublas;
  matrix_column<mapped_matrix<double>> col = column(*matrix_ptr, feature_index);
  std::vector<double> vec;
  for (unsigned i = 0; i < col.size(); ++i) {
    vec.push_back(col[i]);
  }
  std::sort(vec.begin(), vec.end());
  std::vector<std::pair<double, float>> mid_histogram;
  for (unsigned int i = 0; i < FLAGS_histogram_bin_count; ++i) {
    int index = vec.size() * (i + 1) / FLAGS_histogram_bin_count - 1;  // Zero based
    index = index < 0 ? 0 : index; 
    double value = vec[index];
    float portion = (index + 1) * 1.0 / vec.size();
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
  for (unsigned long j = 0; j < m_matrix_ptr_->size2(); ++j) {
    std::vector<std::pair<double, float>> tmp_histogram;
    build_histogram(m_matrix_ptr_, j, &tmp_histogram);
    histogram_vec_ptr->push_back(tmp_histogram);
  }
  for (auto iter1 = histogram_vec_ptr->begin(); iter1 != histogram_vec_ptr->end(); ++iter1) {
    for (auto iter2 = iter1->begin(); iter2 != iter1->end(); ++iter2) {
      VLOG(202) << iter1 - histogram_vec_ptr->begin() << "," << iter2 - iter1->begin() << ":"
                << iter2->first << " " << iter2->second;
    }
  }
  m_histogram_vec_ptr_ = histogram_vec_ptr;
  
  std::shared_ptr<PBTree_Node> root =
      std::shared_ptr<PBTree_Node>(new PBTree_Node());
  std::vector<uint64_t> row_index_vec;
  for (unsigned long i = 0; i < m_matrix_ptr_->size2(); ++i) {
    row_index_vec.push_back(i);
  }
  create_node(row_index_vec, 0, &root);

  return true;
}

}  // namespace pbtree