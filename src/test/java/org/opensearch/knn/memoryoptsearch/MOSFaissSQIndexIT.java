/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

/**
 * This is testing Faiss SQ 32x for all possible combinations.
 */
public class MOSFaissSQIndexIT extends AbstractMemoryOptimizedKnnSearchIT {
    // Explicit BQ encoder params to pin x32 tests to binary quantizer (not BBQ).
    // These tests validate MOS off-heap behavior which is specific to the BQ code path.
    private static final String SQ_ENCODER_PARAMS = """
        {"encoder": {"name": "sq", "parameters": {"bits": 1}}}""";

    public void testNonNestedDiskBasedIndexWithIP() {
        doTestNonNestedIndex(
            VectorDataType.FLOAT,
            SQ_ENCODER_PARAMS,
            false,
            SpaceType.INNER_PRODUCT,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x32
        );
    }

    public void testNestedDiskBasedIndexWithIP() {
        doTestNestedIndex(
            VectorDataType.FLOAT,
            SQ_ENCODER_PARAMS,
            SpaceType.INNER_PRODUCT,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x32
        );
    }
}
