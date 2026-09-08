#
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Provides the prebuilt Intel SVS runtime (conda-forge libsvs-runtime, not built from source) for the vendored
# faiss (knn_sandbox_vendor_faiss) and the SVS JNI library. Resolution order:
#   1. -DSVS_RUNTIME_PREFIX=<dir>  : local, already-extracted package prefix (must contain lib/cmake/svs_runtime)
#   2. -DSVS_RUNTIME_URL=<url>     : artifact to download; requires -DSVS_RUNTIME_SHA256=<hex>
#                                    (dev-only bypass: -DSVS_RUNTIME_ALLOW_UNVERIFIED=ON)
#   3. default URL                 : conda-forge linux-64 libsvs-runtime 0.3.0, sha256-pinned below
# Sets SVS_RUNTIME_PREFIX and prepends it to CMAKE_PREFIX_PATH.

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
        # A custom URL must be checksum-pinned like the default one.
        if(DEFINED SVS_RUNTIME_SHA256 AND NOT "${SVS_RUNTIME_SHA256}" STREQUAL "")
            message(STATUS "SVS runtime: fetching ${SVS_RUNTIME_URL} (sha256-pinned)")
            FetchContent_Declare(svs_runtime_pkg URL "${SVS_RUNTIME_URL}" URL_HASH SHA256=${SVS_RUNTIME_SHA256})
        elseif(SVS_RUNTIME_ALLOW_UNVERIFIED)
            message(WARNING
                "SVS runtime: fetching ${SVS_RUNTIME_URL} WITHOUT checksum verification "
                "(SVS_RUNTIME_ALLOW_UNVERIFIED=ON). Dev-only; never use for a release build.")
            FetchContent_Declare(svs_runtime_pkg URL "${SVS_RUNTIME_URL}")
        else()
            message(FATAL_ERROR
                "SVS runtime: a custom -DSVS_RUNTIME_URL requires -DSVS_RUNTIME_SHA256=<hex> to pin the "
                "artifact. To bypass for local development only, pass -DSVS_RUNTIME_ALLOW_UNVERIFIED=ON.")
        endif()
    else()
        message(STATUS "SVS runtime: fetching ${SVS_RUNTIME_CONDA_DEFAULT_URL}")
        FetchContent_Declare(svs_runtime_pkg
            URL "${SVS_RUNTIME_CONDA_DEFAULT_URL}"
            URL_HASH SHA256=${SVS_RUNTIME_CONDA_DEFAULT_SHA256})
    endif()
    FetchContent_MakeAvailable(svs_runtime_pkg)

    # A .conda artifact is a zip holding inner pkg-*.tar.zst archives; extract the payload in place.
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
        "Provide -DSVS_RUNTIME_PREFIX=<dir> (a libsvs-runtime 0.3.0 prefix) or -DSVS_RUNTIME_URL=<artifact> "
        "plus -DSVS_RUNTIME_SHA256=<hex>, passed through gradle via -Psandbox.cmake.args.")
endif()

list(PREPEND CMAKE_PREFIX_PATH "${SVS_RUNTIME_PREFIX}")
find_package(svs_runtime REQUIRED)
