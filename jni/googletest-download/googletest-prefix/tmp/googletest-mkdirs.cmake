# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/ec2-user/k-NN/jni/googletest-src"
  "/home/ec2-user/k-NN/jni/googletest-build"
  "/home/ec2-user/k-NN/jni/googletest-download/googletest-prefix"
  "/home/ec2-user/k-NN/jni/googletest-download/googletest-prefix/tmp"
  "/home/ec2-user/k-NN/jni/googletest-download/googletest-prefix/src/googletest-stamp"
  "/home/ec2-user/k-NN/jni/googletest-download/googletest-prefix/src"
  "/home/ec2-user/k-NN/jni/googletest-download/googletest-prefix/src/googletest-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/ec2-user/k-NN/jni/googletest-download/googletest-prefix/src/googletest-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/ec2-user/k-NN/jni/googletest-download/googletest-prefix/src/googletest-stamp${cfgdir}") # cfgdir has leading slash
endif()
