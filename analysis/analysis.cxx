// #include <unordered_map>
#include "analysis.h"

namespace pbtree {

bool AnalysisManager::analysis_tree_model() {
  std::map<uint64_t, std::string> feature_name_map;
  for (int i = 0; i < m_fam_ptr_->feature_analysis_model_size(); ++i) {
    const FeatureAnalysisModel& fam = m_fam_ptr_->feature_analysis_model(i);
    LOG(INFO) << fam.feature_name() << " offset " << fam.offset();
    if (fam.feature_type() == FeatureAnalysisModel_FeatureType_CATEGORICAL ||
        fam.feature_type() == FeatureAnalysisModel_FeatureType_MULTI_CATEGORICAL) {
      const FeatureAnalysisModel_BucketInfo& bucket_info = fam.bucket_info();
      feature_name_map[fam.offset()] = fam.feature_name() + "_default";
      for (auto iter : bucket_info.bucket_map()) {
        feature_name_map[fam.offset() + iter.second] = fam.feature_name() + "_" + iter.first;
      }
    }
  }
  m_feature_map_ptr_ = std::make_shared<std::map<uint64_t, std::string>>(feature_name_map);
  LOG(INFO) << "feature_name_map.size() = " << feature_name_map.size();
  std::vector<std::pair<const PBTree_Node*, std::string>> node_ptr_vec;
  node_ptr_vec.push_back(std::make_pair(&(m_pbtree_ptr_->tree(0)), std::string("root")));
  uint pos = 0;
  while (pos < node_ptr_vec.size()) {
    uint64_t feature_index = node_ptr_vec[pos].first->split_feature_index();
    auto feature_iter = m_feature_map_ptr_->find(feature_index);
    std::string feature_name = "undefined";
    if (feature_iter != m_feature_map_ptr_->end()) {
      feature_name = feature_iter->second;
    } else {
      LOG(ERROR) << "Feature index " << feature_index << " not found";
    }
    std::string split_value = std::to_string(node_ptr_vec[pos].first->split_feature_value());
    if (node_ptr_vec[pos].first->has_left_child()) {
      const PBTree_Node* tmp_node = &(node_ptr_vec[pos].first->left_child());
      node_ptr_vec.push_back(std::make_pair(tmp_node, feature_name + "<=" + split_value));
    }
    if (node_ptr_vec[pos].first->has_right_child()) {
      const PBTree_Node* tmp_node = &(node_ptr_vec[pos].first->right_child());
      node_ptr_vec.push_back(std::make_pair(tmp_node, feature_name + ">" + split_value));
    }
    LOG(INFO) << "Level " << node_ptr_vec[pos].first->level() << " " << node_ptr_vec[pos].second;
    ++pos;
  }
  return true;
}

bool AnalysisManager::load_pbtree(const std::string& data_path) {
  FILE* fp = fopen(data_path.data(), "r");
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
  std::shared_ptr<PBTree> pbtree_ptr = std::shared_ptr<PBTree>(new PBTree());
  if (!pbtree_ptr->ParseFromArray(buffer.data(), file_size)) {
    LOG(ERROR) << "Parse string failed";
    return false;
  }
  m_pbtree_ptr_ = pbtree_ptr;
  // LOG(INFO) << m_pbtree_ptr_->DebugString();
  return true;
}

bool AnalysisManager::load_fam(const std::string& data_path) {
  if (data_path.empty()) {
    LOG(ERROR) << "Feature analysis model path is empty!";
    return false;
  }
  FILE* fp = fopen(data_path.data(), "r");
  if (fp == nullptr) {
    LOG(ERROR) << "Open feature analysis mode " << data_path << " failed";
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
  // std::shared_ptr<advertiser::demographyestimate::FeatureAnalysisModelVec> fam_ptr =
  //     std::shared_ptr<advertiser::demographyestimate::FeatureAnalysisModelVec>(
  //         new advertiser::demographyestimate::FeatureAnalysisModelVec());
  // FeatureAnalysisModelVec feature_analysis_model_vec;
  std::shared_ptr<FeatureAnalysisModelVec> fam_ptr =
      std::shared_ptr<FeatureAnalysisModelVec>(
          new FeatureAnalysisModelVec());
  if (!fam_ptr->ParseFromArray(buffer.data(), file_size)) {
    LOG(ERROR) << "Parse string failed";
    return false;
  }
  LOG(INFO) << "Feature Analysis Model Vec size is " << fam_ptr->feature_analysis_model_size();
  LOG(INFO) << "First Feature is " << fam_ptr->feature_analysis_model(0).DebugString();
  set_fam_ptr(fam_ptr);
  return true;
}

}  // pbtree
