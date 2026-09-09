#
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Shared build harness for sandbox tenant JNI libraries: hidden visibility, -Wl,--exclude-libs,ALL (the
# isolation guarantee is Linux-only), a private static copy of the JNI marshalling helpers, and an $ORIGIN
# rpath keep each libopensearchknn_<tenant> independent of the built-in libraries. See sandbox/README.md.

# ---------------------------------------------------------------------------------------------------------
# knn_sandbox_add_jni_library(<target>
#     SOURCES <src>... [INCLUDE_DIRS <dir>...] [LINK_LIBRARIES <lib>...] [DEPENDS <target>...]
# )
#
# Defines the tenant's SHARED JNI library with the isolation recipe:
#   * a private STATIC copy of the JNI marshalling helpers (jni/src/jni_util.cpp + commons.cpp), so the
#     tenant library is runtime-independent of the SHARED opensearchknn_util the built-in libraries link;
#   * hidden symbol visibility plus -Wl,--exclude-libs,ALL (Linux), so symbols from statically linked
#     archives stay local to the tenant .so while its JNI entry points remain exported;
#   * BUILD_RPATH/INSTALL_RPATH of $ORIGIN, so a tenant-shipped runtime .so resolves from its own directory;
#   * the repo-common target properties; registered in TARGET_LIBS for consistency with the built-in targets.
# ---------------------------------------------------------------------------------------------------------
function(knn_sandbox_add_jni_library target)
    cmake_parse_arguments(TENANT "" "" "SOURCES;INCLUDE_DIRS;LINK_LIBRARIES;DEPENDS" ${ARGN})
    if(NOT TENANT_SOURCES)
        message(FATAL_ERROR "knn_sandbox_add_jni_library(${target}): SOURCES is required")
    endif()

    # One shared static helpers target for all tenants (created on first use).
    if(NOT TARGET opensearchknn_sandbox_jni_helpers)
        add_library(opensearchknn_sandbox_jni_helpers STATIC
            ${CMAKE_CURRENT_SOURCE_DIR}/src/jni_util.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/commons.cpp)
        set_property(TARGET opensearchknn_sandbox_jni_helpers PROPERTY POSITION_INDEPENDENT_CODE ON)
        target_include_directories(opensearchknn_sandbox_jni_helpers PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            $ENV{JAVA_HOME}/include
            $ENV{JAVA_HOME}/include/${JVM_OS_TYPE})
    endif()

    add_library(${target} SHARED ${TENANT_SOURCES})
    if(TENANT_DEPENDS)
        add_dependencies(${target} ${TENANT_DEPENDS})
    endif()
    target_link_libraries(${target} opensearchknn_sandbox_jni_helpers ${TENANT_LINK_LIBRARIES})
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        $ENV{JAVA_HOME}/include
        $ENV{JAVA_HOME}/include/${JVM_OS_TYPE}
        ${TENANT_INCLUDE_DIRS})
    set_target_properties(${target} PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN")
    if(${CMAKE_SYSTEM_NAME} STREQUAL Linux)
        # Symbols from statically linked archives (the tenant's vendored vector library) stay local so they
        # cannot interpose with the built-in libraries; the JNI entry points remain exported via JNIEXPORT.
        target_link_options(${target} PRIVATE "-Wl,--exclude-libs,ALL")
    endif()
    opensearch_set_common_properties(${target})
    list(APPEND TARGET_LIBS ${target})
    set(TARGET_LIBS "${TARGET_LIBS}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------------------------------------
# knn_sandbox_vendor_faiss(<prefix>
#     GIT_TAG <sha>
#     [GIT_REPOSITORY <url>]              # default: upstream facebookresearch/faiss
#     [CMAKE_ARGS <-DFAISS_...=...>...]   # tenant-specific extras, e.g. an optional faiss feature flag
# )
# Builds the tenant its own upstream faiss (static, PIC, at the pinned commit, no k-NN patches) for the
# tenant to embed privately via knn_sandbox_add_jni_library. Exports to the caller:
#   <prefix>_INCLUDE_DIR   - include dir for <faiss/...> headers
#   <prefix>_LINK_LIB      - IMPORTED static lib target (BLAS/LAPACK/OpenMP supplied on its INTERFACE)
#   <prefix>_EP_TARGET     - the ExternalProject target to pass as DEPENDS
# ---------------------------------------------------------------------------------------------------------
function(knn_sandbox_vendor_faiss prefix)
    include(ExternalProject)
    cmake_parse_arguments(VENDOR "" "GIT_TAG;GIT_REPOSITORY" "CMAKE_ARGS" ${ARGN})
    if(NOT VENDOR_GIT_TAG)
        message(FATAL_ERROR "knn_sandbox_vendor_faiss(${prefix}): GIT_TAG (a pinned commit) is required")
    endif()
    if(NOT VENDOR_GIT_REPOSITORY)
        set(VENDOR_GIT_REPOSITORY "https://github.com/facebookresearch/faiss.git")
    endif()

    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "(aarch64|arm64|ARM64)")
        set(_vendor_opt_level generic)
        set(_vendor_variant faiss)
    elseif(AVX512_SPR_ENABLED)
        set(_vendor_opt_level avx512_spr)
        set(_vendor_variant faiss_avx512_spr)
    elseif(AVX512_ENABLED)
        set(_vendor_opt_level avx512)
        set(_vendor_variant faiss_avx512)
    elseif(AVX2_ENABLED)
        set(_vendor_opt_level avx2)
        set(_vendor_variant faiss_avx2)
    else()
        set(_vendor_opt_level generic)
        set(_vendor_variant faiss)
    endif()

    set(_vendor_prefix   "${CMAKE_BINARY_DIR}/${prefix}")
    set(_vendor_src      "${_vendor_prefix}/src/${prefix}_ep")
    set(_vendor_build    "${_vendor_prefix}/src/${prefix}_ep-build")
    set(_vendor_lib_path "${_vendor_build}/faiss/${CMAKE_STATIC_LIBRARY_PREFIX}${_vendor_variant}${CMAKE_STATIC_LIBRARY_SUFFIX}")

    ExternalProject_Add(${prefix}_ep
        GIT_REPOSITORY    "${VENDOR_GIT_REPOSITORY}"
        GIT_TAG           "${VENDOR_GIT_TAG}"
        GIT_SHALLOW       FALSE
        PREFIX            "${_vendor_prefix}"
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DBUILD_SHARED_LIBS=OFF
            -DBUILD_TESTING=OFF
            -DFAISS_ENABLE_GPU=OFF
            -DFAISS_ENABLE_PYTHON=OFF
            -DFAISS_ENABLE_C_API=OFF
            -DFAISS_OPT_LEVEL=${_vendor_opt_level}
            ${VENDOR_CMAKE_ARGS}
        INSTALL_COMMAND   ""
        BUILD_BYPRODUCTS  "${_vendor_lib_path}"
    )

    # Embedding the static faiss means re-supplying BLAS/LAPACK/OpenMP on the tenant's link line.
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    find_package(OpenMP REQUIRED)

    add_library(${prefix}_imported STATIC IMPORTED)
    set_target_properties(${prefix}_imported PROPERTIES
        IMPORTED_LOCATION "${_vendor_lib_path}"
        INTERFACE_LINK_LIBRARIES "${BLAS_LIBRARIES};${LAPACK_LIBRARIES};OpenMP::OpenMP_CXX")

    set(${prefix}_INCLUDE_DIR "${_vendor_src}" PARENT_SCOPE)
    set(${prefix}_LINK_LIB "${prefix}_imported" PARENT_SCOPE)
    set(${prefix}_EP_TARGET "${prefix}_ep" PARENT_SCOPE)
endfunction()
