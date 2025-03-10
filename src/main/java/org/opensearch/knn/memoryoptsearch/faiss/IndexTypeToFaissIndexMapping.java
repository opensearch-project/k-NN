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

@UtilityClass
public class IndexTypeToFaissIndexMapping {
    private static final Map<String, Supplier<FaissIndex>> INDEX_TYPE_TO_FAISS_INDEX;

    static {
        final Map<String, Supplier<FaissIndex>> mapping = new HashMap<>();

        mapping.put(FaissIdMapIndex.IXMP, FaissIdMapIndex::new);

        INDEX_TYPE_TO_FAISS_INDEX = Collections.unmodifiableMap(mapping);
    }

    public FaissIndex getFaissIndex(final String indexType) {
        final Supplier<FaissIndex> faissIndexSupplier = INDEX_TYPE_TO_FAISS_INDEX.get(indexType);
        if (faissIndexSupplier != null) {
            return faissIndexSupplier.get();
        }
        throw new UnsupportedFaissIndexException("Index type [" + indexType + "] is not supported.");
    }
}
