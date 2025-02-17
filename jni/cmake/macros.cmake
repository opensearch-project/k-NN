#
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#

macro(opensearch_set_common_properties TARGET)
    set_target_properties(${TARGET} PROPERTIES SUFFIX ${LIB_EXT})
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    if (NOT "${WIN32}" STREQUAL "")
        # Use RUNTIME_OUTPUT_DIRECTORY, to build the target library in the specified directory at runtime.
        set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/release)
    else()
        set_target_properties(${TARGET} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/release)
    endif()
endmacro()
