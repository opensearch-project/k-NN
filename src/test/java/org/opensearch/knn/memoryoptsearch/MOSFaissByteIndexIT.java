/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;

public class MOSFaissByteIndexIT extends AbstractMemoryOptimizedKnnSearchIT {

    public void testNonNestedByteIndexWithL2() {
        doTestNonNestedIndex(VectorDataType.BYTE, EMPTY_PARAMS, false, SpaceType.L2, NO_ADDITIONAL_SETTINGS);
    }

    public void testNestedByteIndexWithL2() {
        doTestNestedIndex(VectorDataType.BYTE, EMPTY_PARAMS, SpaceType.L2, NO_ADDITIONAL_SETTINGS);
    }

    public void testNonNestedByteIndexWithIP() {
        doTestNonNestedIndex(VectorDataType.BYTE, EMPTY_PARAMS, false, SpaceType.INNER_PRODUCT, NO_ADDITIONAL_SETTINGS);
    }

    public void testNestedByteIndexWithIP() {
        doTestNestedIndex(VectorDataType.BYTE, EMPTY_PARAMS, SpaceType.INNER_PRODUCT, NO_ADDITIONAL_SETTINGS);
    }

    public void testWhenNoHNSW() {
        doTestNonNestedIndex(VectorDataType.BYTE, EMPTY_PARAMS, false, SpaceType.L2, NO_BUILD_HNSW);
        doTestNestedIndex(VectorDataType.BYTE, EMPTY_PARAMS, SpaceType.INNER_PRODUCT, NO_BUILD_HNSW);
    }
}
