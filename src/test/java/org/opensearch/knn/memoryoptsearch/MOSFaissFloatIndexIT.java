/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.common.annotation.ExpectRemoteBuildValidation;

public class MOSFaissFloatIndexIT extends AbstractMemoryOptimizedKnnSearchIT {

    @ExpectRemoteBuildValidation
    public void testNonNestedFloatIndexWithL2() {
        doTestNonNestedIndex(VectorDataType.FLOAT, EMPTY_PARAMS, true, SpaceType.L2, NO_ADDITIONAL_SETTINGS);
        doTestNonNestedIndex(VectorDataType.FLOAT, EMPTY_PARAMS, false, SpaceType.L2, NO_ADDITIONAL_SETTINGS);
    }

    @ExpectRemoteBuildValidation
    public void testNestedFloatIndexWithL2() {
        doTestNestedIndex(VectorDataType.FLOAT, EMPTY_PARAMS, SpaceType.L2, NO_ADDITIONAL_SETTINGS);
    }

    @ExpectRemoteBuildValidation
    public void testNonNestedFloatIndexWithIP() {
        doTestNonNestedIndex(VectorDataType.FLOAT, EMPTY_PARAMS, true, SpaceType.INNER_PRODUCT, NO_ADDITIONAL_SETTINGS);
        doTestNonNestedIndex(VectorDataType.FLOAT, EMPTY_PARAMS, false, SpaceType.INNER_PRODUCT, NO_ADDITIONAL_SETTINGS);
    }

    @ExpectRemoteBuildValidation
    public void testNestedFloatIndexWithIP() {
        doTestNestedIndex(VectorDataType.FLOAT, EMPTY_PARAMS, SpaceType.INNER_PRODUCT, NO_ADDITIONAL_SETTINGS);
    }

    public void testWhenNoIndexBuilt() {
        doTestNonNestedIndex(VectorDataType.FLOAT, EMPTY_PARAMS, true, SpaceType.L2, NO_BUILD_HNSW);
        doTestNonNestedIndex(VectorDataType.FLOAT, EMPTY_PARAMS, false, SpaceType.L2, NO_BUILD_HNSW);
    }
}
