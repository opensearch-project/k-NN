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
