/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.Version;
import org.opensearch.knn.common.exception.MemoryOptimizedSearchOldIndicesNotSupportedException;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;

/**
 * This class encapsulates a determination logic for memory optimized search.
 * Memory-optimized-search may not be applied to a certain type of index even {@link KNNEngine} returns a non-null
 * {@link org.opensearch.knn.memoryoptsearch.VectorSearcherFactory}.
 * The overall logic will be made based on the given method context and quantization configuration.
 */
public class MemoryOptimizedSearchSupportSpec {
    private static final Version MIN_VERSION_SUPPORTS_MEM_OPT_SEARCH = Version.V_2_17_0;

    /**
     * Determines whether a memory optimized searching should be applied during search.
     * Note that even when `memory_optimized_search` is not enabled, it will enable memory optimized searching for `on_disk` mode
     * with 1x compression.
     *
     * @param fieldType Field type
     * @param indexName Name of the index
     * @return True if memory optimized search should be used otherwise False.
     */
    public static boolean isSupportedFieldType(final KNNVectorFieldType fieldType, final String indexName) {
        // If the field is configured to always use memory optimized search, return true
        if (fieldType.isAlwaysUseMemoryOptimizedSearch()) {
            return true;
        }

        if (fieldType.isMemoryOptimizedSearchAvailable()) {
            if (KNNSettings.isMemoryOptimizedKnnSearchModeEnabled(indexName)) {
                final boolean shouldBlockMemoryOptimizedSearch = fieldType.getIndexCreatedVersion() == null
                    || fieldType.getIndexCreatedVersion().before(MIN_VERSION_SUPPORTS_MEM_OPT_SEARCH);
                if (shouldBlockMemoryOptimizedSearch) {
                    // Memory-optimized search is enabled, but some existing indices were created before
                    // the minimum version that supports this feature. Throw an exception to clearly
                    // notify the user of the incompatibility.
                    throw new MemoryOptimizedSearchOldIndicesNotSupportedException(
                        "Memory optimized search does not support old indices created before "
                            + MIN_VERSION_SUPPORTS_MEM_OPT_SEARCH.toString()
                            + ". Index ["
                            + indexName
                            + "] was created in "
                            + fieldType.getIndexCreatedVersion().toString()
                    );
                }

                return true;
            }

            // Even mem_opt_srch was disabled, we still enable this for on_disk mode with 1x compression.
            return fieldType.getResolvedSpec().requiresMemoryOptimizedSearchForOnDisk();
        }

        return false;
    }
}
