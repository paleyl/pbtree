#!/bin/sh
wget https://dl.bintray.com/boostorg/release/1.75.0/source/boost_1_75_0.tar.gz
tar xzf boost_1_75_0.tar.gz
git clone https://github.com/google/glog.git
git clone https://github.com/gflags/gflags.git
git clone https://github.com/google/googletest.git
git clone https://github.com/protocolbuffers/protobuf.git

