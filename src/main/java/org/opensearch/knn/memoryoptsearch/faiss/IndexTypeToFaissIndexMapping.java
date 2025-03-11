/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.experimental.UtilityClass;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;

/**
 * This table maintains a mapping between FAISS index types and their corresponding index implementations.
 * A FAISS index file consists of multiple sections, each representing a specific index type.
 * Each section must begin with a unique four-byte string that identifies the index type.
 * This table maps those unique index type strings to their respective index implementations.
 */
@UtilityClass
public class IndexTypeToFaissIndexMapping {
    private static final Map<String, Supplier<FaissIndex>> INDEX_TYPE_TO_FAISS_INDEX;

    static {
        final Map<String, Supplier<FaissIndex>> mapping = new HashMap<>();

        mapping.put(FaissIdMapIndex.IXMP, FaissIdMapIndex::new);

        INDEX_TYPE_TO_FAISS_INDEX = Collections.unmodifiableMap(mapping);
    }

    /**
     * Get index implementation with the given `indexType`. If it fails to find an implementation,
     * it will throw {@link UnsupportedFaissIndexException}
     * @param indexType Unique four bytes index type string.
     * @return Actual implementation that is corresponding to the given index type.
     */
    public FaissIndex getFaissIndex(final String indexType) {
        final Supplier<FaissIndex> faissIndexSupplier = INDEX_TYPE_TO_FAISS_INDEX.get(indexType);
        if (faissIndexSupplier != null) {
            return faissIndexSupplier.get();
        }
        throw new UnsupportedFaissIndexException("Index type [" + indexType + "] is not supported.");
    }
}
