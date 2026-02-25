/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;

/**
 * This is testing pure binary cases where user ingest quantized binary vectors then query with a binary vector while LuceneOnFaiss is on.
 */
public class MOSFaissBinaryIndexIT extends AbstractMemoryOptimizedKnnSearchIT {

    public void testNonNestedBinaryIndexWithHamming() {
        doTestNonNestedIndex(VectorDataType.BINARY, EMPTY_PARAMS, false, SpaceType.HAMMING, NO_ADDITIONAL_SETTINGS);
    }

    public void testNestedBinaryIndexWithHamming() {
        doTestNestedIndex(VectorDataType.BINARY, EMPTY_PARAMS, SpaceType.HAMMING, NO_ADDITIONAL_SETTINGS);
    }

    public void testWhenNoHNSW() {
        doTestNonNestedIndex(VectorDataType.BINARY, EMPTY_PARAMS, false, SpaceType.HAMMING, NO_BUILD_HNSW);
        doTestNestedIndex(VectorDataType.BINARY, EMPTY_PARAMS, SpaceType.HAMMING, NO_BUILD_HNSW);
    }
}
