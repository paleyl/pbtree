// tree
#ifndef TREE_TREE_H_
#define TREE_TREE_H_

#include "float.h"

#include "stdint.h"
#include <string>
#include <vector>
#include <memory>

#include "glog/logging.h"
#include "gflags/gflags.h"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include "boost/numeric/ublas/matrix_sparse.hpp"
#include "boost/numeric/ublas/io.hpp"

#include "distribution/distribution.h"
#include "utility/utility.h"
#include "Tree.pb.h"

namespace pbtree {

class Tree {
 public:
  bool build_tree();

  bool predict(
      const boost::numeric::ublas::matrix_row<
      boost::numeric::ublas::compressed_matrix<double>>& record,
      double* p1, double* p2, double* p3);

  bool predict_one_tree(
      const boost::numeric::ublas::matrix_row<
      boost::numeric::ublas::compressed_matrix<double>>& record,
      const PBTree_Node& root, double* p1, double* p2, double* p3);

  bool create_node(const std::vector<uint64_t>& record_index_vec,
      const uint32_t& level,
      PBTree_Node* node);

  bool build_histogram(
      const std::shared_ptr<boost::numeric::ublas::compressed_matrix<double>>& matrix_ptr,
      const uint64_t feature_index,
      const boost::numeric::ublas::compressed_matrix<double>::iterator1& feature_iter,
      std::vector<std::pair<double, float>>* histogram); 

  bool find_all_feature_split(
      const std::vector<uint64_t>& row_index_vec,
      uint64_t* split_feature_index, double* split_point,
      double* split_loss);

  bool find_one_feature_split(
      const std::vector<uint64_t>& record_index_vec,
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

 private:
  std::shared_ptr<PBTree> m_pbtree_ptr_;
  std::shared_ptr<boost::numeric::ublas::compressed_matrix<double>> m_matrix_ptr_;
  std::shared_ptr<std::vector<double>> m_label_data_ptr_;
  std::shared_ptr<std::vector
      <std::vector<std::pair<double, float>>>> m_histogram_vec_ptr_;
  std::shared_ptr<Distribution> m_distribution_ptr_;
};
}  // namespace pbtree

#endif  // TREE_TREE_H_
