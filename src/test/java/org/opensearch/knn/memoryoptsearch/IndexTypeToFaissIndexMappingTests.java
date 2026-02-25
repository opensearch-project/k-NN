/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIdMapIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;
import org.opensearch.knn.memoryoptsearch.faiss.IndexTypeToFaissIndexMapping;
import org.opensearch.knn.memoryoptsearch.faiss.UnsupportedFaissIndexException;

public class IndexTypeToFaissIndexMappingTests extends KNNTestCase {
    public void testSupportedIndexMapping() {
        // Try to get a supported index.
        final FaissIndex faissIndex = IndexTypeToFaissIndexMapping.getFaissIndex(FaissIdMapIndex.IXMP);
        assertNotNull(faissIndex);
    }

    public void testUnsupportedIndexMapping() {
        // Try to get un-supported index
        try {
            IndexTypeToFaissIndexMapping.getFaissIndex("NOT_SUPPORTED_INDEX_TYPE");
            fail();
        } catch (UnsupportedFaissIndexException e) {}
    }
}
