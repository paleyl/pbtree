#include "float.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "glog/logging.h"
#include "gflags/gflags.h"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include "boost/numeric/ublas/matrix_sparse.hpp"
#include "boost/numeric/ublas/io.hpp"

#include "io/data_manager.h"
#include "distribution/distribution.h"
#include "proto/build/Tree.pb.h"
#include "tree/tree.h"
#include "utility/utility.h"

DEFINE_string(input_train_data_path, "", "");
DEFINE_string(output_model_path, "", "");
DEFINE_string(input_model_path, "", "");
DEFINE_string(input_test_data_path, "", "");
DEFINE_string(running_mode, "train", "");

int blas_trial () {
  using namespace boost::numeric::ublas;
  compressed_matrix<double> m (3, 3, 3 * 3);
  for (unsigned i = 0; i < m.size1 (); ++ i)
      for (unsigned j = 0; j < m.size2 () - 1; ++ j)
          m (i, j) = 3 * i + j;
  VLOG(100) << m;
  auto r1 = m.begin1();
  VLOG(100) << "r1.index1 " << r1.index1() << " r1.index2 " << r1.index2();
  VLOG(100) << *(r1.begin());
  matrix_row<compressed_matrix<double>> ma = row(m, 2);
  // VLOG(102) << "ma value size = " << ma.end() - ma.begin()
  //           << "ma full size = " << ma.size();
  // std::sort(ma.begin(), ma.end());
  std::vector<double> vec;
  for (unsigned i = 0; i < ma.size(); ++i) {
    vec.push_back(ma[i]);
  }
  std::sort(vec.begin(), vec.end(), [](double a, double b) {return a < b;});
  for (auto it = vec.begin(); it != vec.end(); ++it) {
    VLOG(100) << *it;
  }
  //VLOG(100) << *(r1.begin());
  return 0;
}

bool run_test() {
  std::shared_ptr<boost::numeric::ublas::compressed_matrix<double>> feature_matrix_ptr;
  pbtree::DataManager data_manager;
  if (!data_manager.read_train_data(
      FLAGS_input_test_data_path, (uint32_t)0, nullptr, &feature_matrix_ptr)) {
    LOG(ERROR) << "Read test data failed";
    return false;
  }
  FILE* fp = fopen(FLAGS_input_model_path.data(), "r");
  if (fp == nullptr) {
    LOG(ERROR) << "Open input model file failed";
    return false;
  }
  fseek(fp, 0, SEEK_END);
  uint64_t file_size = ftell(fp);
  rewind(fp);
  std::vector<char> buffer(file_size + 10);
  if (!fread(buffer.data(), sizeof(char), file_size, fp)) {
    LOG(ERROR) << "Read data failed";
    return false;
  }
  std::shared_ptr<pbtree::PBTree> pbtree_ptr = std::shared_ptr<pbtree::PBTree>(new pbtree::PBTree());
  if (!pbtree_ptr->ParseFromArray(buffer.data(), file_size)) {
    LOG(ERROR) << "Parse string failed";
    return false;
  }
  LOG(INFO) << pbtree_ptr->DebugString();
  pbtree::Tree tree;
  tree.set_matrix_ptr(feature_matrix_ptr);
  tree.set_pbtree(pbtree_ptr);
  for (unsigned int row_index = 0; row_index < feature_matrix_ptr->size1(); ++row_index) {
    boost::numeric::ublas::matrix_row<
        boost::numeric::ublas::compressed_matrix<double>> record =
            boost::numeric::ublas::row(*feature_matrix_ptr, row_index);
    double p1, p2, p3;
    tree.predict(record, &p1, &p2, &p3);
    VLOG(101) << row_index << " " << p1 << " " << p2 << " " << p3;
  }

  return true;
}

bool run_train() {
  // label_data_t train_data;
  std::vector<double> label_vec;
  std::shared_ptr<boost::numeric::ublas::compressed_matrix<double>> feature_matrix_ptr;
  pbtree::DataManager data_manager;
  if (!data_manager.read_train_data(
      FLAGS_input_train_data_path, (uint32_t)1, &label_vec, &feature_matrix_ptr)) {
    LOG(ERROR) << "Read train data failed";
    return false;
  }

  VLOG(202) << *feature_matrix_ptr;

  VLOG(11) << "Data size is " << feature_matrix_ptr->size1() << "," << feature_matrix_ptr->size2();

  std::shared_ptr<std::vector<double>> label_vec_ptr =
      std::make_shared<std::vector<double>>(label_vec);
  std::shared_ptr<pbtree::Distribution> distribution_ptr =
      std::shared_ptr<pbtree::Distribution>(new pbtree::NormalDistribution());
  pbtree::PBTree pbtree;
  pbtree::Tree tree;
  std::shared_ptr<pbtree::PBTree> pbtree_ptr = std::make_shared<pbtree::PBTree>(pbtree);
  tree.set_matrix_ptr(feature_matrix_ptr);
  tree.set_pbtree(pbtree_ptr);
  tree.set_distribution_ptr(distribution_ptr);
  tree.set_label_data_ptr(label_vec_ptr);
  tree.build_tree();
  LOG(INFO) << pbtree_ptr->DebugString();
  std::string model_output_str;
  pbtree_ptr->SerializeToString(&model_output_str);
  
  std::ofstream fout(FLAGS_output_model_path.data(), std::ios::out | std::ios::binary);
  fout << model_output_str;
  fout.close();
  // for (unsigned int row_index = 0; row_index < feature_matrix_ptr->size1(); ++row_index) {
  //   boost::numeric::ublas::matrix_row<
  //       boost::numeric::ublas::compressed_matrix<double>> record =
  //           boost::numeric::ublas::row(*feature_matrix_ptr, row_index);
  //   double p1, p2, p3;
  //   tree.predict(record, &p1, &p2, &p3);
  //   VLOG(102) << row_index << " " << p1 << " " << p2 << " " << p3;
  // }
  // build_tree(train_data, feature_matrix_ptr, &pbtree);
  // pbtree::PBTree_Node* node = pbtree.add_tree();
  // node->set_level(0);
  // LOG(INFO) << pbtree.DebugString();
  // std::string out;
  // pbtree.SerializeToString(&out);
  // blas_trial();
  return true;
}

int main (int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  if (FLAGS_running_mode == "train") {
    run_train();
  } else if (FLAGS_running_mode == "test") {
    run_test();
  }
  
  // blas_trial();
  return 0;
}
