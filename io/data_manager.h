// data_manager
#ifndef DATA_MANAGER_H_
#define DATA_MANAGER_H_

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

#include "utility/utility.h"

namespace pbtree {

typedef std::pair<uint64_t, double> feature_t;
typedef std::vector<feature_t> featurevec_t;
typedef std::pair<double, featurevec_t> instance_t;
typedef std::vector<instance_t> label_data_t;
typedef std::vector<featurevec_t> predict_data_t;

class DataManager {
 public:
  bool read_train_data(
      const std::string& data_path,
      const uint32_t& major_type,
      std::vector<double>* label_vec,
      std::shared_ptr<boost::numeric::ublas::compressed_matrix<double>>* matrix);

  // bool read_test_data(
  //     const std::string& data_path,
  //     const uint32_t& major_type,
  //     std::shared_ptr<boost::numeric::ublas::compressed_matrix<double>>* matrix);
};

}  // pbtree

#endif  // DATA_MANAGER_H_
