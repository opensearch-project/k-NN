/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;

public class MOSFaissFP16IndexIT extends AbstractMemoryOptimizedKnnSearchIT {
    private static final String METHOD_PARAMETERS = """
        {
          "encoder": {
            "name": "sq",
            "parameters": {
              "type": "fp16"
            }
          }
        }""".trim();

    public void testNonNestedFloatIndexWithL2() {
        doTestNonNestedIndex(VectorDataType.FLOAT, METHOD_PARAMETERS, true, SpaceType.L2, NO_ADDITIONAL_SETTINGS);
        doTestNonNestedIndex(VectorDataType.FLOAT, METHOD_PARAMETERS, false, SpaceType.L2, NO_ADDITIONAL_SETTINGS);
    }

    public void testNestedFloatIndexWithL2() {
        doTestNestedIndex(VectorDataType.FLOAT, METHOD_PARAMETERS, SpaceType.L2, NO_ADDITIONAL_SETTINGS);
    }

    public void testNonNestedFloatIndexWithIP() {
        doTestNonNestedIndex(VectorDataType.FLOAT, METHOD_PARAMETERS, true, SpaceType.INNER_PRODUCT, NO_ADDITIONAL_SETTINGS);
        doTestNonNestedIndex(VectorDataType.FLOAT, METHOD_PARAMETERS, false, SpaceType.INNER_PRODUCT, NO_ADDITIONAL_SETTINGS);
    }

    public void testNestedFloatIndexWithIP() {
        doTestNestedIndex(VectorDataType.FLOAT, METHOD_PARAMETERS, SpaceType.INNER_PRODUCT, NO_ADDITIONAL_SETTINGS);
    }

    public void testNonNestedFloatIndexWithCosine() {
        doTestNonNestedIndex(VectorDataType.FLOAT, METHOD_PARAMETERS, true, SpaceType.COSINESIMIL, NO_ADDITIONAL_SETTINGS);
        doTestNonNestedIndex(VectorDataType.FLOAT, METHOD_PARAMETERS, false, SpaceType.COSINESIMIL, NO_ADDITIONAL_SETTINGS);
    }

    public void testNestedFloatIndexWithCosine() {
        doTestNestedIndex(VectorDataType.FLOAT, METHOD_PARAMETERS, SpaceType.COSINESIMIL, NO_ADDITIONAL_SETTINGS);
    }

    public void testWhenNoIndexBuilt() {
        doTestNonNestedIndex(VectorDataType.FLOAT, METHOD_PARAMETERS, true, SpaceType.L2, NO_BUILD_HNSW);
        doTestNonNestedIndex(VectorDataType.FLOAT, METHOD_PARAMETERS, false, SpaceType.L2, NO_BUILD_HNSW);
    }
}
