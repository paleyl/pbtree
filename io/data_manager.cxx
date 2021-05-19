#include "data_manager.h"

DEFINE_int32(input_data_line_width, 4096, "");

namespace pbtree {

std::string instance_to_str(const instance_t& instance) {
  std::string str;
  str = str + std::to_string(instance.first);
  for (auto iter : instance.second) {
    str = str + " " + std::to_string(iter.first) + "," + std::to_string(iter.second);
  }
  return str;
}

/**
 * @brief  
 * @note   
 * @param  data_path: Input data path.
 * @param  major_type: 0 for row_major, 1 for column_major.
 * @param  label_vec: Pointer to label vector, input nullptr if the dataset has no label.
 * @param  matrix: Pointer to smart pointer of feature matrix.
 * @retval 
 */

bool build_instance_batch(
    std::vector<std::pair<std::pair<uint64_t, uint64_t>, double>>& tmp_data_vec,
    uint32_t batch_index
    ) {
  return true;
}

bool DataManager::read_train_data(
    const std::string& data_path, const uint32_t& major_type,
    std::vector<double>* label_vec,
    std::shared_ptr<boost::numeric::ublas::compressed_matrix<double>>* matrix,
    std::shared_ptr<boost::numeric::ublas::mapped_matrix<double,
    boost::numeric::ublas::row_major, std::unordered_map<std::size_t, double>>>* mapped_matrix /*=nullptr*/) {
  FILE* fp1 = fopen(data_path.data(), "r");
  if (fp1 == nullptr) {
    LOG(ERROR) << "Open input data failed! " << data_path;
    return false;
  }
  char line[FLAGS_input_data_line_width];
  label_data_t data;
  data.clear();
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
    double label = 0;
    uint64_t stating_point = 0;
    if (label_vec != nullptr) {
      label = std::stod(tmp_fields[0]);
      stating_point = 1;
    }
    std::vector<std::string> tmp_pair;
    for (uint64_t i = stating_point; i < tmp_fields.size(); ++i) {
      if (tmp_fields[i].empty()) {
        continue;
      }
      pbtree::Utility::split_string(tmp_fields[i], ":", &tmp_pair);
      uint64_t tmp_index = std::stol(tmp_pair[0]);
      double tmp_value = std::stod(tmp_pair[1]);
      tmp_featurevec.push_back(std::make_pair(tmp_index, tmp_value));
      feature_max_ind = std::max(feature_max_ind, tmp_index);
      ++feature_val_num;
    }
    data.push_back(std::make_pair(label, tmp_featurevec));
  }
  fclose(fp1);

  VLOG(11) << "Data size is " << data.size();
  VLOG(11) << "Sample instance " << data.size() / 2
            << " is: " << instance_to_str(data[data.size() / 2]);

  std::vector<std::pair<std::pair<uint64_t, uint64_t>, double>> tmp_data_vec;
  tmp_data_vec.reserve(feature_val_num);
  for (unsigned long i = 0; i < data.size(); ++i) {
    for (unsigned int j = 0; j < (data)[i].second.size(); ++j) {
      uint64_t feature_index = (data)[i].second[j].first;
      double val = (data)[i].second[j].second;
      VLOG(202) << i << "," << feature_index << ":" << val;
      tmp_data_vec.push_back(std::make_pair(std::make_pair(i, feature_index), val));
    }
  }
  VLOG(101) << "Tmp data size = " << tmp_data_vec.size();
  std::shared_ptr<boost::numeric::ublas::compressed_matrix<double>> mat_ptr;
  if (major_type == 1) {
    boost::numeric::ublas::compressed_matrix<double>
        mat(feature_max_ind + 1, data.size(), feature_val_num + 1);
    mat_ptr =
        std::make_shared<boost::numeric::ublas::compressed_matrix<double>>(mat);
  } else {
    boost::numeric::ublas::compressed_matrix<double>
        mat(data.size(), feature_max_ind + 1, feature_val_num + 1);
    mat_ptr =
        std::make_shared<boost::numeric::ublas::compressed_matrix<double>>(mat);  
  }
  
  uint64_t record_cnt = data.size();
  if (major_type == 0) {  // row_major
    std::sort(tmp_data_vec.begin(), tmp_data_vec.end(),
        [](const std::pair<std::pair<uint64_t, uint64_t>, double>& a,
        const std::pair<std::pair<uint64_t, uint64_t>, double>& b){
          return a.first.first < b.first.first;
        });
    for (auto iter = tmp_data_vec.begin(); iter != tmp_data_vec.end(); ++iter) {
      (*mat_ptr)(iter->first.first, iter->first.second) = iter->second;
    }
  } else if (major_type == 1) {  // column_major
    std::sort(tmp_data_vec.begin(), tmp_data_vec.end(),
        [&](const std::pair<std::pair<uint64_t, uint64_t>, double>& a,
        const std::pair<std::pair<uint64_t, uint64_t>, double>& b){
          return a.first.second * record_cnt + a.first.first < b.first.second * record_cnt + b.first.first;  // sort by column index
        });
    for (auto iter = tmp_data_vec.begin(); iter != tmp_data_vec.begin() + 1000; ++iter) {
      VLOG(101) << iter->first.second << " " << iter->first.first << " " << iter->second;
    }
    for (auto iter = tmp_data_vec.begin(); iter != tmp_data_vec.end(); ++iter) {
      if ((iter - tmp_data_vec.begin()) % 100000 == 0) {
        VLOG(101) << "Built " << iter - tmp_data_vec.begin() << " elements";
      }
      (*mat_ptr)(iter->first.second, iter->first.first) = iter->second;
    }
  }
  *matrix = mat_ptr;
  if (mapped_matrix) {
    boost::numeric::ublas::mapped_matrix<double,
        boost::numeric::ublas::row_major, std::unordered_map<std::size_t, double>> tmp_mapped_matrix(
            mat_ptr->size1(), mat_ptr->size2(), mat_ptr->nnz());
    for (auto iter = tmp_data_vec.begin(); iter != tmp_data_vec.end(); ++iter) {
      tmp_mapped_matrix(iter->first.second, iter->first.first) = iter->second;
    }
    *mapped_matrix = std::make_shared<boost::numeric::ublas::mapped_matrix<double,
        boost::numeric::ublas::row_major, std::unordered_map<std::size_t, double>>>(tmp_mapped_matrix);
  }
  LOG(INFO) << "Input data size1 = " << mat_ptr->size1()
            << " size2 = " << mat_ptr->size2()
            << " nnz = " << mat_ptr->nnz();
  if (label_vec != nullptr) {
    for (auto iter = data.begin(); iter != data.end(); ++iter) {
      label_vec->push_back(iter->first);
    }
  }
  return true;
}

}