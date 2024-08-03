/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.enums;

import org.opensearch.knn.KNNTestCase;

public class QuantizationTypeTests extends KNNTestCase {

    public void testQuantizationTypeValues() {
        QuantizationType[] expectedValues = { QuantizationType.SPACE, QuantizationType.VALUE };
        assertArrayEquals(expectedValues, QuantizationType.values());
    }

    public void testQuantizationTypeValueOf() {
        assertEquals(QuantizationType.SPACE, QuantizationType.valueOf("SPACE"));
        assertEquals(QuantizationType.VALUE, QuantizationType.valueOf("VALUE"));
    }
}
