// analysis
#ifndef ANALYSIS_ANALYSIS_H_
#define ANALYSIS_ANALYSIS_H_

#include "float.h"

#include <algorithm>
#include <cmath>
#include <fstream>
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

#include "FeatureAnalysisModel.pb.h"
#include "Tree.pb.h"
#include "utility/utility.h"

namespace pbtree {

// In this class feature_analysis_model is shorted as fam
class AnalysisManager {
 public:
  bool load_fam(const std::string& data_path);

  bool load_pbtree(const std::string& data_path);

  bool save_model(const std::string& data_path);

  bool analysis_tree_model();

  bool plot_tree_model();

  bool draw_one_node(
      const PBTree_Node& node, const uint32_t& parent_node_id,
      const std::string& parent_split_condition,
      std::string* output_str, uint32_t* node_id);

  bool draw_one_tree(
      const PBTree_Node& node, const uint32_t& tree_index,
      std::string* output_str);

  // bool explore_node(const PBTree_Node& node);

  void set_fam_ptr(
      std::shared_ptr<FeatureAnalysisModelVec> fam_ptr) {
//      std::shared_ptr<advertiser::demographyestimate::FeatureAnalysisModelVec> fam_ptr) {
    m_fam_ptr_ = fam_ptr;
  }

 private:
  std::shared_ptr<FeatureAnalysisModelVec> m_fam_ptr_;
//  std::shared_ptr<advertiser::demographyestimate::FeatureAnalysisModelVec> m_fam_ptr_;
  std::shared_ptr<PBTree> m_pbtree_ptr_;
  std::shared_ptr<std::map<uint64_t, std::string>> m_feature_map_ptr_;
};

}  // pbtree

#endif  // ANALYSIS_ANALYSIS_H_
