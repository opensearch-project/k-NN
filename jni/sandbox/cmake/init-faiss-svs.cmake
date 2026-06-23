#
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Build a SEPARATE, SVS-enabled Faiss for the isolated SVS JNI library.
#
# This is intentionally NOT the k-NN faiss submodule (jni/external/faiss, pinned at the stock main
# revision and built FAISS_ENABLE_SVS=OFF). The experimental SVS tenant tracks LATEST UPSTREAM Faiss,
# which carries the maintained SVS surface (Vamana, IVF-SVS, LeanVec, impl/svs_io) and the current
# find_package(svs_runtime) model. It is built in its own binary tree so its `faiss*` CMake targets
# never collide with the main add_subdirectory(external/faiss) build, and so the SVS .so embeds its own
# faiss object code (kept private via hidden visibility + the version script).
#
# Requires fetch-svs-runtime.cmake to have run first (sets SVS_RUNTIME_PREFIX + svs::svs_runtime).
#
# Exports to the caller:
#   SVS_FAISS_INCLUDE_DIR  - include dir for <faiss/...> headers (the upstream source tree)
#   SVS_FAISS_LINK_LIB     - IMPORTED static lib target to link into libopensearchknn_svs
#   faiss_svs              - ExternalProject target the SVS lib must depend on (add_dependencies)

include(ExternalProject)

# Pinned to 67f066f7a: an SVS-capable upstream Faiss commit (carries IndexSVSVamana, the SVS factory
# descriptions and the find_package(svs_runtime) model). The source is built EXACTLY as upstream ships
# it — no k-NN patches are applied. The SVS JNI wrapper (jni/sandbox/src/svs_wrapper.cpp) deliberately
# uses only upstream faiss APIs so this can hold.
set(SVS_FAISS_GIT_REPOSITORY "https://github.com/facebookresearch/faiss.git"
    CACHE STRING "Upstream Faiss repo for the SVS tenant build")
set(SVS_FAISS_GIT_TAG "67f066f7a02f76d3178baccf4c31b4839ff0fee8"
    CACHE STRING "Pinned SVS-capable upstream Faiss commit for the SVS tenant build")

# Pick the host opt-level / faiss variant the same way init-faiss.cmake does, so the SVS lib links a
# variant matching the CPU it will run on.
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "(aarch64|arm64|ARM64)")
    set(_svs_faiss_opt_level generic)
    set(_svs_faiss_variant faiss)
elseif(AVX512_SPR_ENABLED)
    set(_svs_faiss_opt_level avx512_spr)
    set(_svs_faiss_variant faiss_avx512_spr)
elseif(AVX512_ENABLED)
    set(_svs_faiss_opt_level avx512)
    set(_svs_faiss_variant faiss_avx512)
elseif(AVX2_ENABLED)
    set(_svs_faiss_opt_level avx2)
    set(_svs_faiss_variant faiss_avx2)
else()
    set(_svs_faiss_opt_level generic)
    set(_svs_faiss_variant faiss)
endif()

set(_svs_faiss_prefix   "${CMAKE_BINARY_DIR}/faiss-svs")
set(_svs_faiss_src      "${_svs_faiss_prefix}/src/faiss_svs")
set(_svs_faiss_build    "${_svs_faiss_prefix}/src/faiss_svs-build")
# faiss emits its static libs under <build>/faiss/.
set(_svs_faiss_lib_path "${_svs_faiss_build}/faiss/${CMAKE_STATIC_LIBRARY_PREFIX}${_svs_faiss_variant}${CMAKE_STATIC_LIBRARY_SUFFIX}")

ExternalProject_Add(faiss_svs
    GIT_REPOSITORY    "${SVS_FAISS_GIT_REPOSITORY}"
    GIT_TAG           "${SVS_FAISS_GIT_TAG}"
    GIT_SHALLOW       FALSE
    PREFIX            "${_svs_faiss_prefix}"
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DBUILD_SHARED_LIBS=OFF
        -DBUILD_TESTING=OFF
        -DFAISS_ENABLE_GPU=OFF
        -DFAISS_ENABLE_PYTHON=OFF
        -DFAISS_ENABLE_C_API=OFF
        -DFAISS_ENABLE_SVS=ON
        -DFAISS_OPT_LEVEL=${_svs_faiss_opt_level}
        -DCMAKE_PREFIX_PATH=${SVS_RUNTIME_PREFIX}
    # We only need the static libs; faiss has no usable install for a static, isolated embed here.
    INSTALL_COMMAND   ""
    BUILD_BYPRODUCTS  "${_svs_faiss_lib_path}"
)

set(SVS_FAISS_INCLUDE_DIR "${_svs_faiss_src}" CACHE PATH "SVS-tenant Faiss include dir" FORCE)

# Imported target for the variant static lib. faiss links svs::svs_runtime / BLAS / LAPACK / OpenMP
# transitively; since we embed the static .a we must re-supply those on the SVS lib's link line.
# svs::svs_runtime and OpenMP are added on the libopensearchknn_svs target in jni/CMakeLists.txt;
# BLAS/LAPACK are added here as INTERFACE deps of the imported target.
# VERIFY (build loop): confirm the variant static lib name/path and the full transitive link closure
# (MKL vs OpenBLAS) on the chosen build host; faiss CI uses MKL (BLA_VENDOR=Intel10_64_dyn).
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

add_library(${TARGET_LIB_SVS}_faiss_imported STATIC IMPORTED)
set_target_properties(${TARGET_LIB_SVS}_faiss_imported PROPERTIES
    IMPORTED_LOCATION "${_svs_faiss_lib_path}"
    INTERFACE_LINK_LIBRARIES "${BLAS_LIBRARIES};${LAPACK_LIBRARIES}")

set(SVS_FAISS_LINK_LIB "${TARGET_LIB_SVS}_faiss_imported" CACHE STRING "SVS-tenant Faiss link target" FORCE)
