#include "tree.h"

DEFINE_int32(histogram_bin_count, 100, "number of bins in histogram");

namespace pbtree {

bool Tree::find_split(
    const std::vector<uint64_t>& row_index_vec, const uint64_t& col_index, double* split_point) {
  const std::vector<std::pair<double, float>>& histogram = (*m_histogram_vec_ptr_)[col_index];
  for (auto iter = histogram.begin(); iter != histogram.end(); ++iter) {
    // double point = iter->first;
    
  }
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
  for (int i = 0; i < FLAGS_histogram_bin_count; ++i) {
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

// bool build_tree_node() {

// }

bool Tree::build_tree() {
  PBTree_Node* root_node = m_pbtree_ptr_->add_tree();
  root_node->set_level(0);
  root_node->set_mu(0);
  root_node->set_sigma(0);
  std::vector<double> gain_vec;
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

  return true;
}

}  // namespace pbtree