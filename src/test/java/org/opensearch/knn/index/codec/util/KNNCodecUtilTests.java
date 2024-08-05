/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import junit.framework.TestCase;
import org.opensearch.knn.index.VectorDataType;

import static org.opensearch.knn.index.codec.util.KNNCodecUtil.calculateArraySize;

public class KNNCodecUtilTests extends TestCase {

    public void testCalculateArraySize() {
        int numVectors = 4;
        int vectorLength = 10;

        // Float data type
        VectorDataType vectorDataType = VectorDataType.FLOAT;
        assertEquals(160, calculateArraySize(numVectors, vectorLength, vectorDataType));

        // Byte data type
        vectorDataType = VectorDataType.BYTE;
        assertEquals(40, calculateArraySize(numVectors, vectorLength, vectorDataType));

        // Binary data type
        vectorDataType = VectorDataType.BINARY;
        assertEquals(40, calculateArraySize(numVectors, vectorLength, vectorDataType));
    }
}
