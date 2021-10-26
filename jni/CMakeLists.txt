#
#   SPDX-License-Identifier: Apache-2.0
#
#   The OpenSearch Contributors require contributions made to
#   this file be licensed under the Apache-2.0 license or a
#   compatible open source license.
#
#   Modifications Copyright OpenSearch Contributors. See
#   GitHub history for details.
#
#
#   Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   A copy of the License is located at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   or in the "license" file accompanying this file. This file is distributed
#   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#   express or implied. See the License for the specific language governing
#   permissions and limitations under the License.
#

cmake_minimum_required(VERSION 3.17)

project(KNNPlugin_JNI)

# ---------------------------------- SETUP ----------------------------------
# Target library to be compiled
set(TARGET_LIB OpensearchKNN)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set OS specific variables
if (${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
    set(CMAKE_MACOSX_RPATH 1)
    set(JVM_OS_TYPE darwin)
    set(LIB_EXT .jnilib)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL Linux)
    set(JVM_OS_TYPE linux)
    set(LIB_EXT .so)
else()
    message(FATAL_ERROR "Unable to run on system: ${CMAKE_SYSTEM_NAME}")
endif()

if(NOT KNN_PLUGIN_VERSION)
    set(KNN_PLUGIN_VERSION "1.2.0.0")
endif()

# Set architecture specific variables
if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL aarch64)
    set(MACH_ARCH arm64)
elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL x86_64)
    set(MACH_ARCH x64)
endif()
# ----------------------------------------------------------------------------

# ---------------------------------- NMSLIB ----------------------------------
# Check if nmslib exists
find_path(NMS_REPO_DIR NAMES similarity_search PATHS ${CMAKE_CURRENT_SOURCE_DIR}/external/nmslib)

# If not, pull the updated submodule
if (NOT EXISTS ${NMS_REPO_DIR})
    message(STATUS "Could not find nmslib. Pulling updated submodule.")
    execute_process(COMMAND git submodule update --init -- external/nmslib WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif ()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/nmslib/similarity_search)
# ---------------------------------------------------------------------------

# ---------------------------------- FAISS ----------------------------------
# Avoid building faiss tests
set(BUILD_TESTING OFF)
set(BLA_STATIC ON)
set(FAISS_OPT_LEVEL generic)

if (${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY /usr/local/opt/libomp/lib/libomp.dylib)
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY /usr/local/opt/libomp/lib/libomp.dylib)
    endif()
endif()

find_package(OpenMP REQUIRED)
find_package(ZLIB REQUIRED)
find_package(BLAS REQUIRED)
enable_language(Fortran)
find_package(LAPACK REQUIRED)

# Check if faiss exists
find_path(FAISS_REPO_DIR NAMES faiss PATHS ${CMAKE_CURRENT_SOURCE_DIR}/external/faiss)

# If not, pull the updated submodule
if (NOT EXISTS ${FAISS_REPO_DIR})
    message(STATUS "Could not find faiss. Pulling updated submodule.")
    execute_process(COMMAND git submodule update --init -- external/faiss WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif ()

set(FAISS_ENABLE_GPU OFF)
set(FAISS_ENABLE_PYTHON OFF)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/faiss EXCLUDE_FROM_ALL)
# ---------------------------------------------------------------------------

# ------------------------------ Lib Compiling ------------------------------
add_library(${TARGET_LIB} SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/org_opensearch_knn_index_JNIService.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/jni_util.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/nmslib_wrapper.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/faiss_wrapper.cpp)
target_link_libraries(${TARGET_LIB} faiss NonMetricSpaceLib ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} OpenMP::OpenMP_CXX)
target_include_directories(${TARGET_LIB} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include $ENV{JAVA_HOME}/include $ENV{JAVA_HOME}/include/${JVM_OS_TYPE} ${CMAKE_CURRENT_SOURCE_DIR}/external/faiss ${CMAKE_CURRENT_SOURCE_DIR}/external/nmslib/similarity_search/include)

set_target_properties(${TARGET_LIB} PROPERTIES SUFFIX ${LIB_EXT})
set_target_properties(${TARGET_LIB} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${TARGET_LIB} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/release)
# ---------------------------------------------------------------------------

# --------------------------------- TESTS -----------------------------------
# Reference - https://crascit.com/2015/07/25/cmake-gtest/
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download"
        )
execute_process(COMMAND "${CMAKE_COMMAND}" --build .
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download"
        )
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

add_subdirectory("${CMAKE_BINARY_DIR}/googletest-src"
        "${CMAKE_BINARY_DIR}/googletest-build"
        )
add_executable(
        jni_test
        tests/faiss_wrapper_test.cpp
        tests/nmslib_wrapper_test.cpp
        tests/test_util.cpp)

target_link_libraries(
        jni_test
        gtest_main
        gmock_main
        faiss
        NonMetricSpaceLib
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        OpenMP::OpenMP_CXX
        ${TARGET_LIB}
)

target_include_directories(jni_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        $ENV{JAVA_HOME}/include
        $ENV{JAVA_HOME}/include/${JVM_OS_TYPE}
        ${CMAKE_CURRENT_SOURCE_DIR}/external/faiss
        ${CMAKE_CURRENT_SOURCE_DIR}/external/nmslib/similarity_search/include
        ${gtest_SOURCE_DIR}/include
        ${gmock_SOURCE_DIR}/include)


set_target_properties(jni_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# ---------------------------------------------------------------------------

# -------------------------------- INSTALL ----------------------------------
# Installation rules for shared library
install(TARGETS ${TARGET_LIB}
        LIBRARY DESTINATION lib
        COMPONENT library)

set(KNN_MAINTAINER "OpenSearch Team <opensearch@amazon.com>")
set(OPENSEARCH_DOWNLOAD_URL "https://opensearch.org/downloads.html")
set(CPACK_PACKAGE_NAME ${KNN_PACKAGE_NAME})
set(CPACK_PACKAGE_VERSION ${KNN_PLUGIN_VERSION})
set(CMAKE_INSTALL_PREFIX /usr)
set(CPACK_GENERATOR "RPM;DEB")
set(CPACK_OUTPUT_FILE_PREFIX packages)
set(CPACK_PACKAGE_RELEASE 1)
set(CPACK_PACKAGE_VENDOR "Amazon")
set(CPACK_PACKAGE_CONTACT "Maintainer: ${KNN_MAINTAINER}")
set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
set(CPACK_COMPONENTS_GROUPING IGNORE)
get_cmake_property(CPACK_COMPONENTS_ALL COMPONENTS)
list(REMOVE_ITEM CPACK_COMPONENTS_ALL "Unspecified")

# Component variable
set(KNN_PACKAGE_NAME opensearch-knnlib)
set(KNN_PACKAGE_DESCRIPTION "KNN JNI library built off of nmslib and faiss for OpenSearch")

# RPM
set(CPACK_RPM_PACKAGE_LICENSE "ASL-2.0")
set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_RPM_PACKAGE_URL ${OPENSEARCH_DOWNLOAD_URL})
set(CPACK_RPM_PACKAGE_RELEASE ${CPACK_PACKAGE_RELEASE})

set(CPACK_RPM_PACKAGE_NAME ${KNN_PACKAGE_NAME})
set(CPACK_RPM_FILE_NAME "${CPACK_RPM_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${JVM_OS_TYPE}-${MACH_ARCH}.rpm")
set(CPACK_RPM_PACKAGE_DESCRIPTION ${KNN_PACKAGE_DESCRIPTION})
set(CPACK_RPM_PACKAGE_SUMMARY "OpenSearch k-NN JNI Library with nmslib and faiss")

# DEB
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE ${OPENSEARCH_DOWNLOAD_URL})
set(CPACK_DEBIAN_PACKAGE_MAINTAINER ${KNN_MAINTAINER})
set(CPACK_DEBIAN_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION})
set(CPACK_DEBIAN_PACKAGE_SECTION "libs")
set(CPACK_DEB_COMPONENT_INSTALL ON)

set(CPACK_DEBIAN_PACKAGE_NAME ${KNN_PACKAGE_NAME})
set(CPACK_DEBIAN_FILE_NAME "${CPACK_DEBIAN_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${JVM_OS_TYPE}-${MACH_ARCH}.deb")
set(CPACK_DEBIAN_DESCRIPTION ${KNN_PACKAGE_DESCRIPTION})
set(CPACK_DEBIAN_PACKAGE_SOURCE ${CPACK_DEBIAN_PACKAGE_NAME})

include(CPack)
# ---------------------------------------------------------------------------
