#!/bin/sh
#sudo yum install gcc-c++ libstdc++-devel git cmake
wget wget https://boostorg.jfrog.io/artifactory/main/release/1.75.0/source/boost_1_75_0.tar.gz
tar xzf boost_1_75_0.tar.gz
git clone https://github.com/google/glog.git
cd glog;git checkout c8f8135a5720aee7de8328b42e4c43f8aa2e60aa;mkdir build;cd build;cmake ..;make -j2;cd ../../
git clone https://github.com/gflags/gflags.git
cd gflags;git checkout 827c769e5fc98e0f2a34c47cef953cc6328abced;mkdir build;cd build;cmake ..;make -j2;cd ../../
git clone https://github.com/google/googletest.git
cd googletest;git checkout 609281088cfefc76f9d0ce82e1ff6c30cc3591e5;mkdir build;cd build;cmake ..;make -j2;cd ../../
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf;git checkout aee143afe8c17a3d2c7e88b70ffa6e08a73e2683;mkdir build;cd build;cmake ..;make -j2;cd ../../
git clone https://github.com/gperftools/gperftools.git
cd gperftools;git checkout f7c6fb6c8e99d6b1b725e5994373bcd19ffdf8fd;mkdir build;cd build;cmake ..;make -j2;cd ../../
