cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(PBTREE VERSION 1.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_C_COMPILER g++)
set(CMAKE_CXX_COMPILER g++)
SET(CMAKE_CXX_FLAGS "-O2 -g -Wall -Werror")
SET(CMAKE_EXE_LINKER_FLAGS "-static-libgcc  -static-libstdc++")

list(APPEND EXTRA_LIBS "${PROJECT_SOURCE_DIR}/thirdparty/gflags/build/lib/libgflags.a")
list(APPEND EXTRA_LIBS "${PROJECT_SOURCE_DIR}/thirdparty/glog/build/libglog.a")
list(APPEND EXTRA_LIBS "${PROJECT_SOURCE_DIR}/thirdparty/gperftools/.libs/libtcmalloc_and_profiler.a")
list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/thirdparty/gflags/build/include")
list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/thirdparty/glog/build")
list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/thirdparty/glog/src")
list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/thirdparty/protobuf/src/")
#add_subdirectory(proto)
#list(APPEND EXTRA_LIBS "${PROJECT_SOURCE_DIR}/build/proto")
list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/thirdparty/boost_1_75_0/")
list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/thirdparty/gperftools/src")

add_subdirectory(analysis)
list(APPEND EXTRA_LIBS analysis)
add_subdirectory(distribution)
list(APPEND EXTRA_LIBS distribution)
add_subdirectory(tree)
list(APPEND EXTRA_LIBS tree)
add_subdirectory(io)
list(APPEND EXTRA_LIBS io)
add_subdirectory(utility)
list(APPEND EXTRA_LIBS utility)
## It seemed appeded later will be linked first, pb is depended by others so should be in the latest
add_subdirectory(proto)
list(APPEND EXTRA_LIBS tree_proto)
#list(APPEND EXTRA_LIBS "${PROJECT_SOURCE_DIR}/proto/build/libtree_proto.a")

add_executable(pbtree pbtree.cxx)

target_link_libraries(pbtree PUBLIC -lpthread)
target_link_libraries(pbtree PUBLIC ${EXTRA_LIBS})
target_link_libraries(pbtree PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/protobuf/cmake/build/libprotobuf.a")

target_include_directories(pbtree PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           "${PROJECT_SOURCE_DIR}/build/proto"
                           ${CMAKE_CURRENT_SOURCE_DIR}
                           ${EXTRA_INCLUDES}
                           ${PROTOBUF_LIBRARY}
                           )
