#
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#

# Check if nmslib exists
find_path(NMS_REPO_DIR NAMES similarity_search PATHS ${CMAKE_CURRENT_SOURCE_DIR}/external/nmslib NO_DEFAULT_PATH)

# If not, pull the updated submodule
if (NOT EXISTS ${NMS_REPO_DIR})
    message(STATUS "Could not find nmslib. Pulling updated submodule.")
    execute_process(COMMAND git submodule update --init -- external/nmslib WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif ()

# Check if patch exist, this is to skip git apply during CI build. See CI.yml with ubuntu.
find_path(PATCH_FILE NAMES 0001-Initialize-maxlevel-during-add-from-enterpoint-level.patch PATHS ${CMAKE_CURRENT_SOURCE_DIR}/patches/nmslib NO_DEFAULT_PATH)

# If it exists, apply patches
if (EXISTS ${PATCH_FILE})
    message(STATUS "Applying custom patches.")
    execute_process(COMMAND git ${GIT_PATCH_COMMAND} --3way --ignore-space-change --ignore-whitespace ${CMAKE_CURRENT_SOURCE_DIR}/patches/nmslib/0001-Initialize-maxlevel-during-add-from-enterpoint-level.patch WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external/nmslib ERROR_VARIABLE ERROR_MSG RESULT_VARIABLE RESULT_CODE)

    if(RESULT_CODE)
        message(FATAL_ERROR "Failed to apply patch:\n${ERROR_MSG}")
    endif()
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/nmslib/similarity_search)
