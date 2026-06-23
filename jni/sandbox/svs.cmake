#
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Build logic for the experimental, fully-isolated Intel SVS JNI library (libopensearchknn_svs).
# Included from jni/CMakeLists.txt only when -DCONFIG_SVS=ON (driven by -Pknn.sandbox.enabled=true).
# All SVS-specific native code lives under jni/sandbox/; the only sources reused from jni/src are the
# generic, faiss-free JNI marshalling helpers (jni_util + commons), compiled in as a static copy so the
# library stays an independent artifact. It embeds its own UNMODIFIED upstream SVS-enabled faiss (no k-NN
# patches) and exports only its JNI symbols (see svs_jni.version).
#
# CMAKE_CURRENT_SOURCE_DIR here is the jni/ root (include() does not change it).
set(SVS_SANDBOX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sandbox)

# Provide the SVS runtime (conda-forge libsvs-runtime) and build the SVS-enabled upstream faiss.
# Sets: svs::svs_runtime, SVS_FAISS_LINK_LIB, SVS_FAISS_INCLUDE_DIR, and the faiss_svs ExternalProject.
include(${SVS_SANDBOX_DIR}/cmake/fetch-svs-runtime.cmake)
include(${SVS_SANDBOX_DIR}/cmake/init-faiss-svs.cmake)
find_package(OpenMP REQUIRED)

# STATIC marshalling helpers (jni_util + commons), embedded into the SVS lib so it carries its own copy
# and stays runtime-independent of the SHARED opensearchknn_util the other libs link.
add_library(${TARGET_LIB_JNI_HELPERS} STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/jni_util.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/commons.cpp)
set_property(TARGET ${TARGET_LIB_JNI_HELPERS} PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(${TARGET_LIB_JNI_HELPERS} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include $ENV{JAVA_HOME}/include $ENV{JAVA_HOME}/include/${JVM_OS_TYPE})

add_library(
    ${TARGET_LIB_SVS} SHARED
    ${SVS_SANDBOX_DIR}/src/org_opensearch_knn_sandbox_svs_SvsService.cpp
    ${SVS_SANDBOX_DIR}/src/svs_wrapper.cpp
    ${SVS_SANDBOX_DIR}/src/svs_constants.cpp
)
add_dependencies(${TARGET_LIB_SVS} faiss_svs)
target_link_libraries(${TARGET_LIB_SVS} ${SVS_FAISS_LINK_LIB} ${TARGET_LIB_JNI_HELPERS} svs::svs_runtime OpenMP::OpenMP_CXX)
target_compile_definitions(${TARGET_LIB_SVS} PRIVATE FAISS_ENABLE_SVS FAISS_SVS_RUNTIME_VERSION=v0)
target_include_directories(${TARGET_LIB_SVS} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${SVS_SANDBOX_DIR}/include
    $ENV{JAVA_HOME}/include
    $ENV{JAVA_HOME}/include/${JVM_OS_TYPE}
    ${SVS_FAISS_INCLUDE_DIR}
)
# Hidden visibility + version script: export ONLY the JNI entry points so this lib's embedded faiss
# symbols cannot interpose with the (different-version) faiss embedded in libopensearchknn_faiss when
# both are dlopen'd into the same JVM. $ORIGIN rpath lets it find the bundled libsvs_runtime.so beside it.
set_target_properties(${TARGET_LIB_SVS} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN")
if (${CMAKE_SYSTEM_NAME} STREQUAL Linux)
    target_link_options(${TARGET_LIB_SVS} PRIVATE
        "-Wl,--version-script=${SVS_SANDBOX_DIR}/cmake/svs_jni.version")
endif()
opensearch_set_common_properties(${TARGET_LIB_SVS})
list(APPEND TARGET_LIBS ${TARGET_LIB_SVS})
