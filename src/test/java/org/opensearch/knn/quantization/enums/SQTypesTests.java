/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.enums;

import org.opensearch.knn.KNNTestCase;

public class SQTypesTests extends KNNTestCase {
    public void testSQTypesValues() {
        ScalarQuantizationType[] expectedValues = {
            ScalarQuantizationType.ONE_BIT,
            ScalarQuantizationType.TWO_BIT,
            ScalarQuantizationType.FOUR_BIT,
            ScalarQuantizationType.UNSUPPORTED_TYPE };
        assertArrayEquals(expectedValues, ScalarQuantizationType.values());
    }

    public void testSQTypesValueOf() {
        assertEquals(ScalarQuantizationType.ONE_BIT, ScalarQuantizationType.valueOf("ONE_BIT"));
        assertEquals(ScalarQuantizationType.TWO_BIT, ScalarQuantizationType.valueOf("TWO_BIT"));
        assertEquals(ScalarQuantizationType.FOUR_BIT, ScalarQuantizationType.valueOf("FOUR_BIT"));
        assertEquals(ScalarQuantizationType.UNSUPPORTED_TYPE, ScalarQuantizationType.valueOf("UNSUPPORTED_TYPE"));
    }
}
