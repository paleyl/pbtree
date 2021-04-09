#include "float.h"

#include <algorithm>
#include <cmath>
#include <iostream>
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

#include "distribution/distribution.h"
#include "proto/build/Tree.pb.h"
#include "tree/tree.h"
#include "utility/utility.h"

DEFINE_string(test, "xxx", "");
DEFINE_int32(input_data_line_width, 4096, "");
DEFINE_string(input_train_data_path, "", "");

DEFINE_double(default_value, 0.0, "");

typedef std::pair<uint64_t, double> feature_t;
typedef std::vector<feature_t> featurevec_t;
typedef std::pair<double, featurevec_t> instance_t;
typedef std::vector<instance_t> label_data_t;
typedef std::vector<featurevec_t> predict_data_t;

bool read_data(
    const std::string& data_path,
    label_data_t* data,
    std::shared_ptr<boost::numeric::ublas::mapped_matrix<double>>* matrix) {
  FILE* fp1 = fopen(data_path.data(), "r");
  if (fp1 == nullptr) {
    LOG(ERROR) << "Open input data failed! " << data_path;
    return false;
  }
  char line[FLAGS_input_data_line_width];
  data->clear();
  uint64_t feature_max_ind = 0;
  uint64_t feature_val_num = 0;
  while (!feof(fp1) && fgets(line, sizeof(line), fp1) != nullptr) {
    instance_t tmp_instance;
    featurevec_t tmp_featurevec;
    std::vector<std::string> tmp_fields;
    if (!pbtree::Utility::split_string(std::string(line), " ", &tmp_fields)) {
      LOG(ERROR) << "Split failed: " << line;
      return false;
    }
    double label = std::stod(tmp_fields[0]);
    std::vector<std::string> tmp_pair;
    for (uint64_t i = 1; i < tmp_fields.size(); ++i) {
      pbtree::Utility::split_string(tmp_fields[i], ":", &tmp_pair);
      uint64_t tmp_index = std::stol(tmp_pair[0]);
      double tmp_value = std::stod(tmp_pair[1]);
      tmp_featurevec.push_back(std::make_pair(tmp_index, tmp_value));
      feature_max_ind = std::max(feature_max_ind, tmp_index);
      ++feature_val_num;
    }
    data->push_back(std::make_pair(label, tmp_featurevec));
  }
  boost::numeric::ublas::mapped_matrix<double>
      mat(data->size(), feature_max_ind + 1, feature_val_num + 1);
  auto mat_ptr =
      std::make_shared<boost::numeric::ublas::mapped_matrix<double>>(mat);
  for (unsigned long i = 0; i < data->size(); ++i) {
    for (unsigned int j = 0; j < (*data)[i].second.size(); ++j) {
      uint64_t ind = (*data)[i].second[j].first;
      double val = (*data)[i].second[j].second;
      VLOG(202) << i << "," << ind << ":" << val;
      (*mat_ptr)(i, ind) = val;
    }
  }
  *matrix = mat_ptr;
  return true;
}

std::string instance_to_str(const instance_t& instance) {
  std::string str;
  str = str + std::to_string(instance.first);
  for (auto iter : instance.second) {
    str = str + " " + std::to_string(iter.first) + "," + std::to_string(iter.second);
  }
  return str;
}

int blas_trial () {
  using namespace boost::numeric::ublas;
  mapped_matrix<double> m (3, 3, 3 * 3);
  for (unsigned i = 0; i < m.size1 (); ++ i)
      for (unsigned j = 0; j < m.size2 () - 1; ++ j)
          m (i, j) = 3 * i + j;
  VLOG(100) << m;
  auto r1 = m.begin1();
  VLOG(100) << *(r1.begin());
  matrix_row<mapped_matrix<double>> ma = row(m, 2);
  // std::sort(ma.begin(), ma.end());
  std::vector<double> vec;
  for (unsigned i = 0; i < ma.size(); ++i) {
    vec.push_back(ma[i]);
  }
  std::sort(vec.begin(), vec.end(), [](double a, double b) {return a > b;});
  for (auto it = vec.cbegin(); it != vec.end(); ++it) {
    VLOG(100) << *it;
  }
  //VLOG(100) << *(r1.begin());
  return 0;
}

bool run_test1() {
  return true;
}

bool run_test() {
  label_data_t train_data;
  std::shared_ptr<boost::numeric::ublas::mapped_matrix<double>> feature_matrix_ptr;
  if (!read_data(FLAGS_input_train_data_path, &train_data, &feature_matrix_ptr)) {
    LOG(ERROR) << "Read train data failed";
    return false;
  };
  VLOG(202) << *feature_matrix_ptr;
  VLOG(11) << "Data size is " << train_data.size();
  VLOG(11) << "Sample instance " << train_data.size() / 2
            << " is: " << instance_to_str(train_data[train_data.size() / 2]);
  VLOG(11) << "Data size is " << feature_matrix_ptr->size1() << "," << feature_matrix_ptr->size2();
  std::vector<double> label_data;
  for (auto iter = train_data.begin(); iter < train_data.end(); ++iter) {
    label_data.push_back(iter->first);
  }
  std::shared_ptr<std::vector<double>> label_data_ptr =
      std::make_shared<std::vector<double>>(label_data);
  std::shared_ptr<pbtree::Distribution> distribution_ptr =
      std::shared_ptr<pbtree::Distribution>(new pbtree::NormalDistribution());
  pbtree::PBTree pbtree;  
  pbtree::Tree tree;
  std::shared_ptr<pbtree::PBTree> pbtree_ptr = std::make_shared<pbtree::PBTree>(pbtree);
  tree.set_matrix_ptr(feature_matrix_ptr);
  tree.set_pbtree(pbtree_ptr);
  tree.set_distribution_ptr(distribution_ptr);
  tree.set_label_data_ptr(label_data_ptr);
  tree.build_tree();
  VLOG(102) << pbtree_ptr->DebugString();

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
  run_test();
  // blas_trial();
  return 0;
}
