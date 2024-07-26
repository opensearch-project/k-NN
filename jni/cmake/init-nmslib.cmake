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


# Apply patches
if(NOT DEFINED APPLY_LIB_PATCHES OR "${APPLY_LIB_PATCHES}" STREQUAL true)
    # Define list of patch files
    set(PATCH_FILE_LIST)
    list(APPEND PATCH_FILE_LIST "${CMAKE_CURRENT_SOURCE_DIR}/patches/nmslib/0001-Initialize-maxlevel-during-add-from-enterpoint-level.patch")
    list(APPEND PATCH_FILE_LIST "${CMAKE_CURRENT_SOURCE_DIR}/patches/nmslib/0002-Adds-ability-to-pass-ef-parameter-in-the-query-for-h.patch")

    # Get patch id of the last commit
    execute_process(COMMAND sh -c "git --no-pager show HEAD | git patch-id --stable" OUTPUT_VARIABLE PATCH_ID_OUTPUT_FROM_COMMIT WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external/nmslib)
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
endif()

# Apply patch files
list(SORT PATCH_FILES_TO_APPLY)
foreach(PATCH_FILE IN LISTS PATCH_FILES_TO_APPLY)
    message(STATUS "Applying patch of ${PATCH_FILE}")
    execute_process(COMMAND git ${GIT_PATCH_COMMAND} --3way --ignore-space-change --ignore-whitespace ${PATCH_FILE} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external/nmslib ERROR_VARIABLE ERROR_MSG RESULT_VARIABLE RESULT_CODE)
    if(RESULT_CODE)
        message(FATAL_ERROR "Failed to apply patch:\n${ERROR_MSG}")
    endif()
endforeach()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/nmslib/similarity_search)
