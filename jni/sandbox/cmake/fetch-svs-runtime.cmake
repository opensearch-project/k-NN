#
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Provide the prebuilt Intel SVS runtime (libsvs_runtime.so + its CMake package config) so that both
# the SVS-enabled upstream Faiss (built in init-faiss-svs.cmake) and the SVS JNI library can resolve
# `find_package(svs_runtime)`.
#
# This mirrors how current upstream Faiss obtains the runtime: it is NOT built from source, it is the
# prebuilt conda-forge package `libsvs-runtime` (pinned to 0.3.0 by upstream Faiss in conda/faiss-gpu/
# meta.yaml). The package contains lib/libsvs_runtime.so* and lib/cmake/svs_runtime/svs_runtimeConfig.cmake.
#
# Resolution order:
#   1. -DSVS_RUNTIME_PREFIX=<dir>  : a local, already-extracted package prefix (offline / air-gapped /
#                                    reproducible CI). The dir must contain lib/cmake/svs_runtime.
#   2. -DSVS_RUNTIME_URL=<url>     : a package artifact to download + extract (e.g. a mirrored copy).
#   3. default URL                 : the conda-forge linux-64 libsvs-runtime 0.3.0 artifact (sha256-pinned).
#
# Sets SVS_RUNTIME_PREFIX (cache) to the resolved prefix and prepends it to CMAKE_PREFIX_PATH, then
# runs find_package(svs_runtime REQUIRED) for this (outer) build. init-faiss-svs.cmake forwards
# SVS_RUNTIME_PREFIX to the Faiss ExternalProject so its own find_package resolves the same runtime.

set(SVS_RUNTIME_CONDA_DEFAULT_URL
    "https://anaconda.org/conda-forge/libsvs-runtime/0.3.0/download/linux-64/libsvs-runtime-0.3.0-gc187b54_0_0.conda"
    CACHE STRING "Default conda-forge libsvs-runtime artifact URL")
set(SVS_RUNTIME_CONDA_DEFAULT_SHA256
    "406f34af39beed8d087b424177056cdea4c3b73f2b19d5d3f8b51f3a6fe7d113"
    CACHE STRING "sha256 of the default conda-forge libsvs-runtime artifact")

if(DEFINED SVS_RUNTIME_PREFIX AND NOT "${SVS_RUNTIME_PREFIX}" STREQUAL "")
    message(STATUS "SVS runtime: using local prefix ${SVS_RUNTIME_PREFIX}")
else()
    include(FetchContent)
    if(DEFINED SVS_RUNTIME_URL AND NOT "${SVS_RUNTIME_URL}" STREQUAL "")
        message(STATUS "SVS runtime: fetching ${SVS_RUNTIME_URL}")
        FetchContent_Declare(svs_runtime_pkg URL "${SVS_RUNTIME_URL}")
    else()
        message(STATUS "SVS runtime: fetching ${SVS_RUNTIME_CONDA_DEFAULT_URL}")
        FetchContent_Declare(svs_runtime_pkg
            URL "${SVS_RUNTIME_CONDA_DEFAULT_URL}"
            URL_HASH SHA256=${SVS_RUNTIME_CONDA_DEFAULT_SHA256})
    endif()
    FetchContent_MakeAvailable(svs_runtime_pkg)

    # A plain tarball mirror extracts straight to the package prefix (lib/...). A conda-format `.conda`
    # artifact is a zip holding inner pkg-*.tar.zst archives; extract the payload one into place.
    if(NOT EXISTS "${svs_runtime_pkg_SOURCE_DIR}/lib/cmake/svs_runtime")
        file(GLOB _svs_runtime_inner_pkgs "${svs_runtime_pkg_SOURCE_DIR}/pkg-*.tar.zst")
        foreach(_svs_inner ${_svs_runtime_inner_pkgs})
            message(STATUS "SVS runtime: extracting inner conda payload ${_svs_inner}")
            file(ARCHIVE_EXTRACT INPUT "${_svs_inner}" DESTINATION "${svs_runtime_pkg_SOURCE_DIR}")
        endforeach()
    endif()
    set(SVS_RUNTIME_PREFIX "${svs_runtime_pkg_SOURCE_DIR}" CACHE PATH "Resolved SVS runtime prefix" FORCE)
endif()

if(NOT EXISTS "${SVS_RUNTIME_PREFIX}/lib/cmake/svs_runtime")
    message(FATAL_ERROR
        "SVS runtime package at '${SVS_RUNTIME_PREFIX}' is missing lib/cmake/svs_runtime. "
        "Provide -Psvs.prefix=<dir> (a libsvs-runtime 0.3.0 prefix) or -Psvs.url=<artifact>.")
endif()

list(PREPEND CMAKE_PREFIX_PATH "${SVS_RUNTIME_PREFIX}")
find_package(svs_runtime REQUIRED)
