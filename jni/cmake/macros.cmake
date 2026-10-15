#
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#

option(KNN_ALLOW_NON_RELEASE_BUILD "Allow k-NN JNI libraries to be built with a build type other than Release" OFF)
# Fails compilation of any JNI library not built as Release; see src/knn_build_guard.cpp.
set(KNN_BUILD_GUARD_SOURCE "${CMAKE_CURRENT_LIST_DIR}/../src/knn_build_guard.cpp")
# KNN_RELEASE_BUILD is only defined for the Release configuration (case-insensitive), so Debug, RelWithDebInfo,
# MinSizeRel, None and custom build types all fail the guard unless KNN_ALLOW_NON_RELEASE_BUILD is ON.
set(KNN_BUILD_CONFIG_DEFINITIONS
    "$<$<CONFIG:Release>:KNN_RELEASE_BUILD>"
    "$<$<BOOL:${KNN_ALLOW_NON_RELEASE_BUILD}>:KNN_ALLOW_NON_RELEASE_BUILD>")

macro(opensearch_set_common_properties TARGET)
    set_target_properties(${TARGET} PROPERTIES SUFFIX ${LIB_EXT})
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_sources(${TARGET} PRIVATE ${KNN_BUILD_GUARD_SOURCE})
    target_compile_definitions(${TARGET} PRIVATE ${KNN_BUILD_CONFIG_DEFINITIONS})

    if (NOT "${WIN32}" STREQUAL "")
        # Use RUNTIME_OUTPUT_DIRECTORY, to build the target library in the specified directory at runtime.
        set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/release)
    else()
        set_target_properties(${TARGET} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/release)
    endif()
endmacro()
