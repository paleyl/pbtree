
#include "model_manager.h"

namespace pbtree {

bool ModelManager::load_tree_model(
    const std::string& data_path,
    std::shared_ptr<PBTree>* pbtree_ptr) {
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
  fclose(fp);
  std::shared_ptr<PBTree> local_pbtree_ptr = std::shared_ptr<PBTree>(new PBTree());
  if (!local_pbtree_ptr->ParseFromArray(buffer.data(), file_size)) {
    LOG(ERROR) << "Parse string failed";
    return false;
  }
  VLOG(303) << local_pbtree_ptr->DebugString();
  *pbtree_ptr = local_pbtree_ptr;
  return true;
}

bool ModelManager::save_tree_model(
    const PBTree& pbtree,
    const std::string& data_path) {
  std::string model_output_str;
  pbtree.SerializeToString(&model_output_str);
  std::ofstream fout(data_path.data(), std::ios::out | std::ios::binary);
  fout << model_output_str;
  fout.close();
  return true;
}

bool ModelManager::load_fam(
    const std::string& data_path,
    std::shared_ptr<FeatureAnalysisModelVec>* fam_ptr) {
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
  fclose(fp);

  std::shared_ptr<FeatureAnalysisModelVec> local_fam_ptr =
      std::shared_ptr<FeatureAnalysisModelVec>(
          new FeatureAnalysisModelVec());
  if (!local_fam_ptr->ParseFromArray(buffer.data(), file_size)) {
    LOG(ERROR) << "Parse string failed";
    return false;
  }
  LOG(INFO) << "Feature Analysis Model Vec size is " << local_fam_ptr->feature_analysis_model_size();
  LOG(INFO) << "First Feature is " << local_fam_ptr->feature_analysis_model(0).DebugString();
  *fam_ptr = local_fam_ptr;
  return true;
}

}  // namespace pbtree
