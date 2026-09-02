#
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Discovers and configures the native side of every sandbox tenant. Included from jni/CMakeLists.txt only
# when -DCONFIG_SANDBOX=ON (driven by the Gradle opt-in -Pknn.sandbox.enabled=true); a default build never
# reaches this file. See sandbox/README.md for the onboarding tutorial.
#
# CMAKE_CURRENT_SOURCE_DIR here is the jni/ root (include() does not change it).
set(KNN_SANDBOX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sandbox)

include(${KNN_SANDBOX_DIR}/cmake/SandboxTenant.cmake)

file(GLOB _knn_sandbox_tenant_files "${KNN_SANDBOX_DIR}/*/tenant.cmake")
list(LENGTH _knn_sandbox_tenant_files _knn_sandbox_tenant_count)
message(STATUS "Sandbox: CONFIG_SANDBOX=ON, ${_knn_sandbox_tenant_count} native tenant(s) discovered")
foreach(_knn_sandbox_tenant_file ${_knn_sandbox_tenant_files})
    get_filename_component(_knn_sandbox_tenant_dir "${_knn_sandbox_tenant_file}" DIRECTORY)
    get_filename_component(KNN_SANDBOX_TENANT_NAME "${_knn_sandbox_tenant_dir}" NAME)
    set(KNN_SANDBOX_TENANT_DIR "${_knn_sandbox_tenant_dir}")
    message(STATUS "Sandbox: configuring tenant '${KNN_SANDBOX_TENANT_NAME}'")
    include(${_knn_sandbox_tenant_file})
    if(NOT TARGET opensearchknn_${KNN_SANDBOX_TENANT_NAME})
        message(FATAL_ERROR "jni/sandbox/${KNN_SANDBOX_TENANT_NAME}/tenant.cmake must define target opensearchknn_${KNN_SANDBOX_TENANT_NAME}")
    endif()
endforeach()
