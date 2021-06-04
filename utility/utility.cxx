#include "utility.h"

namespace pbtree {
bool Utility::split_string(
    const std::string &src, const char *separators,
    std::vector<std::string> *result) {
  result->clear();
  if (src.empty() || separators == NULL) {
    return false;
  }
  static const int kDictLen = 128;  // 0~127
  char dict[kDictLen] = {'\0'};
  const char *p = separators;
  while (*p) {
    const int pvalue = static_cast<int>(*p);
    if (pvalue >= kDictLen || pvalue < 0) {
      return false;
    } else {
      dict[pvalue] = 1;
    }
    ++p;
  }
  size_t last_pos = 0;
  p = src.c_str();
  for (size_t pos = 0; pos < src.size(); ++pos) {
    const int pvalue = static_cast<int>(p[pos]);
    if (pvalue >= kDictLen || pvalue < 0) {
      continue;
    } else if (dict[static_cast<int>(pvalue)]) {
      result->push_back(src.substr(last_pos, pos-last_pos));
      last_pos = pos+1;
    }
  }
  if (last_pos == src.size()) {
    result->push_back("");
  } else {
    result->push_back(src.substr(last_pos));
  }
  return true;
}

bool Utility::check_double_equal(const double& a, const double& b) {
  if ((a - b > 1e-10) || (b - a) > 1e-10) 
    return false;
  else
    return true;
}

bool Utility::check_double_le(const double& a, const double& b) {  // less or equal
  if (a - b < 1e-50)
    return true;
  else
    return false;
}
// bool read_data(const std::string& data_path, label_data_t* data) {
//   FILE* fp1 = fopen(data_path.data(), "r");
//   if (fp1 == nullptr) {
//     LOG(INFO) << "Open input data failed! " << data_path;
//     return false;
//   }
//   char line[FLAGS_input_data_line_width];
//   data->clear();
//   while (!feof(fp1) && fgets(line, sizeof(line), fp1) != nullptr) {
//     instance_t instance;
//     featurevec_t featurevec;
//     std::vector<std::string> tmp_fields;
//     SplitString(std::string(line), " ", &tmp_fields);
//     double label = std::stod(tmp_fields[0]);
    

//   }

//   return true;
// }
}  // namespace pbtree
