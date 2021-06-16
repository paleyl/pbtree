// tree
#ifndef TREE_TREE_H_
#define TREE_TREE_H_

#include "float.h"

#include "stdint.h"
#include <string>
#include <vector>
#include <memory>
#include <tuple>
#include <unordered_set>

#include "glog/logging.h"
#include "gflags/gflags.h"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include "boost/numeric/ublas/matrix_sparse.hpp"
#include "boost/numeric/ublas/io.hpp"

#include "distribution/distribution.h"
#include "utility/utility.h"
#include "Tree.pb.h"
#include "thread_pool.h"
#include "io/model_manager.h"

namespace pbtree {

class Tree {
 public:
  bool build_tree();

  bool init();

  bool init_pred_dist_vec(
      const double& p1, const double& p2, const double& p3);

  bool predict(
      const boost::numeric::ublas::matrix_row<
      boost::numeric::ublas::compressed_matrix<double>>& record,
      double* p1, double* p2, double* p3);

  bool predict_one_tree(
      const boost::numeric::ublas::matrix_row<
      boost::numeric::ublas::compressed_matrix<double>>& record,
      const PBTree_Node& root, double* p1, double* p2, double* p3);
  
  bool boost_predict_data_set(
      const boost::numeric::ublas::compressed_matrix<double>& matrix,
      std::vector<std::tuple<double, double, double>>* predicted_vec,
      std::vector<std::tuple<double, double>>* pred_moment_vec,
      std::vector<std::pair<double, double>>* pred_interval_vec);

  bool boost_update(const PBTree_Node& new_tree);

  bool boost_update_one_instance(
      const PBTree_Node& new_tree,
      unsigned long record_index,
      double* p1, double* p2, double* p3);

  bool create_node(const std::vector<uint64_t>& record_index_vec,
      const uint32_t& level,
      PBTree_Node* node, const std::vector<uint64_t>* input_feature_vec = nullptr);

  bool build_histogram(
      const std::shared_ptr<boost::numeric::ublas::compressed_matrix<double>>& matrix_ptr,
      const uint64_t feature_index,
      const boost::numeric::ublas::compressed_matrix<double>::iterator1& feature_iter,
      std::vector<std::pair<double, float>>* histogram); 

  bool find_all_feature_split(
      const std::vector<uint64_t>& row_index_vec,
      uint64_t* split_feature_index, double* split_point,
      double* split_loss, const std::vector<uint64_t>* candidate_feature_vec);

  bool find_one_feature_split(
      const std::vector<uint64_t>* record_index_vec,
      const uint64_t& feature_index, double* split_point,
      double* split_loss);

  void set_pbtree(std::shared_ptr<PBTree> pbtree_ptr) {
    m_pbtree_ptr_ = pbtree_ptr;
  }
  void set_matrix_ptr(
      std::shared_ptr<boost::numeric::ublas::compressed_matrix<double>> matrix_ptr) {
    m_matrix_ptr_ = matrix_ptr;
  }
  void set_label_data_ptr(
      std::shared_ptr<std::vector<double>> label_data_ptr) {
    m_label_data_ptr_ = label_data_ptr;
  }
  void set_distribution_ptr(
      std::shared_ptr<Distribution> distribution_ptr) {
    m_distribution_ptr_ = distribution_ptr;
  }

  bool check_split_histogram(const uint64_t& feature_index);

  bool check_valid_candidate(
      const std::vector<uint64_t> record_index_vec,
      const std::vector<uint64_t> previous_feature_vec,
      std::vector<uint64_t>* candidate_feature_vec);

  void do_intersection(
      const std::vector<uint64_t>::const_iterator& iter_begin,
      const std::vector<uint64_t>::const_iterator& iter_end,
      const std::vector<uint64_t>* record_inded_vec_ptr,
      uint64_t* num);

  void do_intersection1(
      const std::vector<uint64_t>* record_index_vec_ptr,
      const std::vector<uint64_t>* pre_filtered_feature_vec_ptr,
      const uint32_t& begin_index,
      const uint32_t& end_index,
      std::vector<uint64_t>* result_vec_ptr);

 private:
  std::shared_ptr<PBTree> m_pbtree_ptr_;
  std::shared_ptr<boost::numeric::ublas::compressed_matrix<double>> m_matrix_ptr_;
  std::shared_ptr<std::vector<double>> m_label_data_ptr_;
  std::shared_ptr<std::vector
      <std::vector<std::pair<double, float>>>> m_histogram_vec_ptr_;  // value, percentile
  std::shared_ptr<Distribution> m_distribution_ptr_;
  std::shared_ptr<std::vector<std::tuple<double, double, double>>> m_pred_param_vec_ptr_;
  std::shared_ptr<ThreadPool> m_thread_pool_ptr_;
  std::shared_ptr<std::vector<uint64_t>> m_valid_split_feature_vec_ptr_;
  std::shared_ptr<std::vector<std::pair<uint64_t, std::vector<std::pair<double, float>>>>> m_valid_histogram_vec_ptr_;
  std::shared_ptr<std::vector<std::pair<uint64_t, std::pair<double, double>>>> m_candidate_split_vec_ptr_;
  std::shared_ptr<std::map<uint64_t, std::vector<uint64_t>>> m_non_zero_value_map_ptr_;
  uint64_t m_max_non_zero_per_feature_;
  // std::shared_ptr<std::unordered_set<uint64_t>> m_candidate_feature_set_ptr_;
};
}  // namespace pbtree

#endif  // TREE_TREE_H_
