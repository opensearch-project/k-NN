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
        doTestNonNestedIndex(VectorDataType.FLOAT, METHOD_PARAMETERS, true, SpaceType.L2);
        doTestNonNestedIndex(VectorDataType.FLOAT, METHOD_PARAMETERS, false, SpaceType.L2);
    }

    public void testNestedFloatIndexWithL2() {
        doTestNestedIndex(VectorDataType.FLOAT, METHOD_PARAMETERS, SpaceType.L2);
    }

    public void testNonNestedFloatIndexWithIP() {
        doTestNonNestedIndex(VectorDataType.FLOAT, METHOD_PARAMETERS, true, SpaceType.INNER_PRODUCT);
        doTestNonNestedIndex(VectorDataType.FLOAT, METHOD_PARAMETERS, false, SpaceType.INNER_PRODUCT);
    }

    public void testNestedFloatIndexWithIP() {
        doTestNestedIndex(VectorDataType.FLOAT, METHOD_PARAMETERS, SpaceType.INNER_PRODUCT);
    }
}
