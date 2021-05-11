// model_manager
#ifndef MODEL_MANAGER_H_
#define MODEL_MANAGER_H_

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
#include "Tree.pb.h"
#include "FeatureAnalysisModel.pb.h"

#include "utility/utility.h"

namespace pbtree {

class ModelManager {
 public:
  bool load_tree_model(
      const std::string& data_path,
      std::shared_ptr<PBTree>* pbtree_ptr);

  bool save_tree_model(
      const PBTree& pbtree,
      const std::string& data_path);

  bool load_fam(
      const std::string& data_path,
      std::shared_ptr<FeatureAnalysisModelVec>* fam_ptr);

};

}

#endif  // MODEL_MANAGER_H_
