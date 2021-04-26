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

#include "analysis/analysis.h"
#include "distribution/distribution.h"
#include "io/data_manager.h"
#include "proto/build/Tree.pb.h"
#include "tree/tree.h"
#include "utility/utility.h"

DEFINE_string(input_analysis_data_path, "", "");
DEFINE_string(input_train_data_path, "", "");
DEFINE_string(output_model_path, "", "");
DEFINE_string(input_model_path, "", "");
DEFINE_string(input_fam_path, "", "");
DEFINE_string(input_test_data_path, "", "");
DEFINE_string(running_mode, "train", "");
DEFINE_string(distribution_type, "GAMMA_DISTRIBUTION", "");

// what to do next, use forest and calculate mixture

bool run_analysis() {
  pbtree::AnalysisManager analysis_manager;
  analysis_manager.load_fam(FLAGS_input_fam_path);
  analysis_manager.load_pbtree(FLAGS_input_model_path);

  pbtree::DataManager data_manager;
  std::vector<double> label_vec;
  std::shared_ptr<boost::numeric::ublas::compressed_matrix<double>> feature_matrix_ptr;
  if (!data_manager.read_train_data(
      FLAGS_input_analysis_data_path, (uint32_t)1, &label_vec, &feature_matrix_ptr)) {
    LOG(ERROR) << "Read analysis data failed";
    return false;
  }
  analysis_manager.set_feature_matrix_ptr(feature_matrix_ptr);
  std::shared_ptr<std::vector<double>> label_vec_ptr =
      std::make_shared<std::vector<double>>(label_vec);
  analysis_manager.set_label_vec_ptr(label_vec_ptr);
  analysis_manager.analysis_tree_model();
  analysis_manager.plot_tree_model();
  return true;
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
  // auto descriptor = pbtree::PBTree_DistributionType_descriptor();
  // auto tmp = descriptor->FindValueByName(FLAGS_distribution_type)->index();
  pbtree::PBTree_DistributionType tmp_type;
  pbtree::PBTree_DistributionType_Parse(FLAGS_distribution_type, &tmp_type);
  std::shared_ptr<pbtree::Distribution> distribution_ptr =
      pbtree::DistributionManager::get_distribution(tmp_type);
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
  return true;
}

int main (int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  if (FLAGS_running_mode == "train") {
    run_train();
  } else if (FLAGS_running_mode == "test") {
    run_test();
  } else if (FLAGS_running_mode == "analysis") {
    run_analysis();
  }
  return 0;
}
