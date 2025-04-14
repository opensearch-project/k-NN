/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;

public class MOSFaissByteIndexIT extends AbstractMemoryOptimizedKnnSearchIT {

    public void testNonNestedFloatIndexWithL2() {
        doTestNonNestedIndex(VectorDataType.BYTE, EMPTY_PARAMS, true, SpaceType.L2);
        doTestNonNestedIndex(VectorDataType.BYTE, EMPTY_PARAMS, false, SpaceType.L2);
    }

    public void testNestedFloatIndexWithL2() {
        doTestNestedIndex(VectorDataType.BYTE, EMPTY_PARAMS, SpaceType.L2);
    }

    public void testNonNestedFloatIndexWithIP() {
        doTestNonNestedIndex(VectorDataType.BYTE, EMPTY_PARAMS, true, SpaceType.INNER_PRODUCT);
        doTestNonNestedIndex(VectorDataType.BYTE, EMPTY_PARAMS, false, SpaceType.INNER_PRODUCT);
    }

    public void testNestedFloatIndexWithIP() {
        doTestNestedIndex(VectorDataType.BYTE, EMPTY_PARAMS, SpaceType.INNER_PRODUCT);
    }
}
