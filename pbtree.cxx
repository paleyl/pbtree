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
// #include "gperftools/profiler.h"

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include "boost/numeric/ublas/matrix_sparse.hpp"
#include "boost/numeric/ublas/io.hpp"

#include "analysis/analysis.h"
#include "distribution/distribution.h"
#include "io/data_manager.h"
#include "io/model_manager.h"
#include "proto/build/Tree.pb.h"
#include "tree/tree.h"
#include "utility/utility.h"

DEFINE_string(input_analysis_data_path, "", "");
DEFINE_string(input_train_data_path, "", "");
// DEFINE_string(output_model_path, "", "");
DEFINE_string(input_model_path, "", "");
DEFINE_string(input_fam_path, "", "");
DEFINE_string(input_test_data_path, "", "");
DEFINE_string(running_mode, "train", "");
DEFINE_string(distribution_type, "GAMMA_DISTRIBUTION", "");
DECLARE_string(output_model_path);
DEFINE_string(output_predict_result_path, "", "");
DEFINE_bool(do_profiling, false, "");

// what to do next, use forest and calculate mixture
// TODO(paleylv): 1. Optimize regularization. 2. Add model manager 
// 3. Add optimizer manager for different optimizing mode

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

bool run_boost_predict() {
  std::shared_ptr<boost::numeric::ublas::compressed_matrix<double>> feature_matrix_ptr;
  pbtree::DataManager data_manager;
  std::vector<double> label_data_vec;
  if (!data_manager.read_train_data(
      FLAGS_input_test_data_path, (uint32_t)1, &label_data_vec, &feature_matrix_ptr)) {
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
  fclose(fp);
  std::shared_ptr<pbtree::PBTree> pbtree_ptr = std::shared_ptr<pbtree::PBTree>(new pbtree::PBTree());
  if (!pbtree_ptr->ParseFromArray(buffer.data(), file_size)) {
    LOG(ERROR) << "Parse string failed";
    return false;
  }
  LOG(INFO) << pbtree_ptr->DebugString();
  pbtree::Tree tree;
  tree.set_matrix_ptr(feature_matrix_ptr);
  tree.set_pbtree(pbtree_ptr);
  std::vector<std::vector<double>> pred_param_vec;
  std::vector<std::tuple<double, double>> pred_moment_vec;
  std::vector<std::pair<double, double>> pred_interval_vec;
  tree.boost_predict_data_set(*feature_matrix_ptr, &pred_param_vec, &pred_moment_vec, &pred_interval_vec);
  CHECK_EQ(pred_param_vec.size(), label_data_vec.size());
  CHECK_EQ(pred_moment_vec.size(), label_data_vec.size());
  uint64_t hit_count = 0;
  double relative_interval_width = 0;
  std::ofstream output_result_file;
  output_result_file.open(FLAGS_output_predict_result_path);
  for (unsigned long i = 0; i < pred_param_vec.size(); ++i) {
    if (pred_interval_vec[i].first < label_data_vec[i] &&
        pred_interval_vec[i].second > label_data_vec[i])
      ++hit_count;
    double tmp_relative_iterval_width =
        (pred_interval_vec[i].second - pred_interval_vec[i].first) / std::get<0>(pred_moment_vec[i]);
    relative_interval_width += tmp_relative_iterval_width;
    LOG(INFO) << i << "-th instance: " << label_data_vec[i]
              << " [" << pred_interval_vec[i].first
              << "," << pred_interval_vec[i].second
              << "), (" 
              << std::get<0>(pred_moment_vec[i]) << ","
              << std::get<1>(pred_moment_vec[i]) << "), ("
              << pred_param_vec[i][0] << ","
              << pred_param_vec[i][1] << ")";
    output_result_file << label_data_vec[i]
                       << "\t" << pred_interval_vec[i].first
                       << "\t" << pred_interval_vec[i].second
                       << "\t" << std::get<0>(pred_moment_vec[i])
                       << "\t" << std::get<1>(pred_moment_vec[i])
                       << "\t" << pred_param_vec[i][0]
                       << "\t" << pred_param_vec[i][1]
                       << "\n";
  }
  output_result_file.close();
  LOG(INFO) << "Total sample = " << pred_param_vec.size()
            << ", hit count = " << hit_count
            << ", hit ratio = " << hit_count * 1.0 / pred_param_vec.size()
            << ", relative interval width = " << relative_interval_width / pred_param_vec.size();
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
  fclose(fp);
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
    std::vector<double> pred_dist;
    tree.predict(record, &pred_dist);
    VLOG(101) << row_index << " " << pred_dist[0] << " " << pred_dist[1];
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
  pbtree::PBTree_DistributionType tmp_type = pbtree::PBTree_DistributionType_GAMMA_DISTRIBUTION;
  pbtree::PBTree_DistributionType_Parse(FLAGS_distribution_type, &tmp_type);
  LOG(INFO) << tmp_type;
  std::shared_ptr<pbtree::Distribution> distribution_ptr =
      pbtree::DistributionManager::get_distribution(tmp_type);
  pbtree::PBTree pbtree;
  pbtree::Tree tree;
  std::shared_ptr<pbtree::PBTree> pbtree_ptr = std::make_shared<pbtree::PBTree>(pbtree);
  tree.set_matrix_ptr(feature_matrix_ptr);
  tree.set_pbtree(pbtree_ptr);
  tree.set_distribution_ptr(distribution_ptr);
  tree.set_label_data_ptr(label_vec_ptr);
  tree.init();
  // tree.init_pred_dist_vec();
  tree.build_tree();
  LOG(INFO) << pbtree_ptr->DebugString();
  std::string model_output_str;
  pbtree_ptr->SerializeToString(&model_output_str);
  
  std::ofstream fout(FLAGS_output_model_path.data(), std::ios::out | std::ios::binary);
  fout << model_output_str;
  fout.close();
  return true;
}

void SignalHandle(const char *data, int size) {
  std::string str = std::string(data, size);
  LOG(ERROR) << str;
}

int main (int argc, char** argv) {

  // ProfilerStart("test.prof");
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  LOG(INFO) << ::google::CommandlineFlagsIntoString();
  ::google::InstallFailureSignalHandler();
  ::google::InstallFailureWriter(&SignalHandle);
  if (FLAGS_running_mode == "train") {
    run_train();
  } else if (FLAGS_running_mode == "test") {
    run_test();
  } else if (FLAGS_running_mode == "analysis") {
    run_analysis();
  } else if (FLAGS_running_mode == "boost_predict") {
    run_boost_predict();
  }

  // ProfilerStop();
  return 0;
}
