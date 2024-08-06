/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.enums;

import org.opensearch.knn.KNNTestCase;

public class ValueQuantizationTypeTests extends KNNTestCase {
    public void testValueQuantizationTypeValues() {
        ValueQuantizationType[] expectedValues = { ValueQuantizationType.SCALAR };
        assertArrayEquals(expectedValues, ValueQuantizationType.values());
    }

    public void testValueQuantizationTypeValueOf() {
        assertEquals(ValueQuantizationType.SCALAR, ValueQuantizationType.valueOf("SCALAR"));
    }
}
