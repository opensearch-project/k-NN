/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common.exception;

import org.opensearch.knn.KNNTestCase;

import java.util.Arrays;
import java.util.List;

public class KNNInvalidIndicesExceptionTests extends KNNTestCase {
    public void testConstructor() {
        List<String> invalidIndices = Arrays.asList("invalid-index-1", "invalid-index-2");
        String message = "test message";
        KNNInvalidIndicesException knnInvalidIndexException = new KNNInvalidIndicesException(invalidIndices, message);
        assertEquals(invalidIndices, knnInvalidIndexException.getInvalidIndices());
        assertEquals(message, knnInvalidIndexException.getMessage());
    }
}
