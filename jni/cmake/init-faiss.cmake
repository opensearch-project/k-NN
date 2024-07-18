#
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#

# Check if faiss exists
find_path(FAISS_REPO_DIR NAMES faiss PATHS ${CMAKE_CURRENT_SOURCE_DIR}/external/faiss NO_DEFAULT_PATH)

# If not, pull the updated submodule
if (NOT EXISTS ${FAISS_REPO_DIR})
    message(STATUS "Could not find faiss. Pulling updated submodule.")
    execute_process(COMMAND git submodule update --init -- external/faiss WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif ()

# Apply patches
if(NOT DEFINED APPLY_LIB_PATCHES OR "${APPLY_LIB_PATCHES}" STREQUAL true)
    # Define list of patch files
    set(PATCH_FILE_LIST)
    list(APPEND PATCH_FILE_LIST "${CMAKE_CURRENT_SOURCE_DIR}/patches/faiss/0001-Custom-patch-to-support-multi-vector.patch")
    list(APPEND PATCH_FILE_LIST "${CMAKE_CURRENT_SOURCE_DIR}/patches/faiss/0002-Enable-precomp-table-to-be-shared-ivfpq.patch")
    list(APPEND PATCH_FILE_LIST "${CMAKE_CURRENT_SOURCE_DIR}/patches/faiss/0003-Custom-patch-to-support-range-search-params.patch")
    list(APPEND PATCH_FILE_LIST "${CMAKE_CURRENT_SOURCE_DIR}/patches/faiss/0004-Custom-patch-to-support-binary-vector.patch")

    # Get patch id of the last commit
    execute_process(COMMAND sh -c "git --no-pager show HEAD | git patch-id --stable" OUTPUT_VARIABLE PATCH_ID_OUTPUT_FROM_COMMIT WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external/faiss)
    string(REPLACE " " ";" PATCH_ID_LIST_FROM_COMMIT ${PATCH_ID_OUTPUT_FROM_COMMIT})
    list(GET PATCH_ID_LIST_FROM_COMMIT 0 PATCH_ID_FROM_COMMIT)

    # Find all patch files need to apply
    list(SORT PATCH_FILE_LIST ORDER DESCENDING)
    set(PATCH_FILES_TO_APPLY)
    foreach(PATCH_FILE IN LISTS PATCH_FILE_LIST)
        # Get patch id of a patch file
        execute_process(COMMAND sh -c "cat ${PATCH_FILE} | git patch-id --stable" OUTPUT_VARIABLE PATCH_ID_OUTPUT)
        string(REPLACE " " ";" PATCH_ID_LIST ${PATCH_ID_OUTPUT})
        list(GET PATCH_ID_LIST 0 PATCH_ID)

        # Add the file to patch list if patch id does not match
        if (${PATCH_ID} STREQUAL ${PATCH_ID_FROM_COMMIT})
            break()
        else()
            list(APPEND PATCH_FILES_TO_APPLY ${PATCH_FILE})
        endif()
    endforeach()

    # Apply patch files
    list(SORT PATCH_FILES_TO_APPLY)
    foreach(PATCH_FILE IN LISTS PATCH_FILES_TO_APPLY)
        message(STATUS "Applying patch of ${PATCH_FILE}")
        execute_process(COMMAND git ${GIT_PATCH_COMMAND} --3way --ignore-space-change --ignore-whitespace ${PATCH_FILE} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external/faiss ERROR_VARIABLE ERROR_MSG RESULT_VARIABLE RESULT_CODE)
        if(RESULT_CODE)
            message(FATAL_ERROR "Failed to apply patch:\n${ERROR_MSG}")
        endif()
    endforeach()
endif()

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

find_package(ZLIB REQUIRED)

# Statically link BLAS - ensure this is before we find the blas package so we dont dynamically link
set(BLA_STATIC ON)
find_package(BLAS REQUIRED)
enable_language(Fortran)
find_package(LAPACK REQUIRED)

# Set relevant properties
set(BUILD_TESTING OFF)          # Avoid building faiss tests
set(FAISS_ENABLE_GPU OFF)
set(FAISS_ENABLE_PYTHON OFF)

if(NOT DEFINED SIMD_ENABLED)
    set(SIMD_ENABLED true)   # set default value as true if the argument is not set
endif()

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm64" OR NOT ${SIMD_ENABLED})
    set(FAISS_OPT_LEVEL generic)    # Keep optimization level as generic on Windows OS as it is not supported due to MINGW64 compiler issue. Also, on aarch64 avx2 is not supported.
    set(TARGET_LINK_FAISS_LIB faiss)
else()
    set(FAISS_OPT_LEVEL avx2)       # Keep optimization level as avx2 to improve performance on Linux and Mac.
    set(TARGET_LINK_FAISS_LIB faiss_avx2)
    string(PREPEND LIB_EXT "_avx2") # Prepend "_avx2" to lib extension to create the library as "libopensearchknn_faiss_avx2.so" on linux and "libopensearchknn_faiss_avx2.jnilib" on mac
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/faiss EXCLUDE_FROM_ALL)
