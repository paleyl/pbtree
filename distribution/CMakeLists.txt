set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_C_COMPILER g++)
set(CMAKE_CXX_COMPILER g++)
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/../")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/boost_1_75_0")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/glog/build")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/glog/src")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/gflags/build/include")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/../build/proto")
add_library(distribution distribution.cxx gamma_distribution.cxx normal_distribution.cxx 
nonparametric_continuous_distribution.cxx 
bayesian_continuous_distribution.cxx 
distribution_utility.cxx)
