#
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Native build for the Intel SVS tenant (libopensearchknn_svs): its own SVS-enabled upstream faiss plus the
# prebuilt Intel SVS runtime.

include(${KNN_SANDBOX_TENANT_DIR}/cmake/fetch-svs-runtime.cmake)

knn_sandbox_vendor_faiss(svs_faiss
    GIT_TAG 124bfa1d4f9f8f3b114bf9da602941d815248e07
    CMAKE_ARGS
        -DFAISS_ENABLE_SVS=ON
        -DCMAKE_PREFIX_PATH=${SVS_RUNTIME_PREFIX}
)

find_package(OpenMP REQUIRED)

knn_sandbox_add_jni_library(opensearchknn_svs
    SOURCES
        ${KNN_SANDBOX_TENANT_DIR}/src/org_opensearch_knn_sandbox_svs_SvsService.cpp
        ${KNN_SANDBOX_TENANT_DIR}/src/svs_wrapper.cpp
        ${KNN_SANDBOX_TENANT_DIR}/src/svs_constants.cpp
    INCLUDE_DIRS
        ${KNN_SANDBOX_TENANT_DIR}/include
        ${svs_faiss_INCLUDE_DIR}
    LINK_LIBRARIES
        ${svs_faiss_LINK_LIB}
        svs::svs_runtime
        OpenMP::OpenMP_CXX
    DEPENDS
        ${svs_faiss_EP_TARGET}
)

target_compile_definitions(opensearchknn_svs PRIVATE FAISS_ENABLE_SVS FAISS_SVS_RUNTIME_VERSION=v0)

add_custom_command(TARGET opensearchknn_svs POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${SVS_RUNTIME_PREFIX}/lib/libsvs_runtime.so.0 ${CMAKE_BINARY_DIR}/release/
    COMMENT "Installing libsvs_runtime beside the tenant JNI library"
)
