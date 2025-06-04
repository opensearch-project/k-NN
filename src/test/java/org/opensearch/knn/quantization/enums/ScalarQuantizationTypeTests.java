/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.enums;

import org.opensearch.knn.KNNTestCase;

import java.util.HashSet;
import java.util.Set;

public class ScalarQuantizationTypeTests extends KNNTestCase {
    public void testSQTypesValues() {
        ScalarQuantizationType[] expectedValues = {
            ScalarQuantizationType.ONE_BIT,
            ScalarQuantizationType.TWO_BIT,
            ScalarQuantizationType.FOUR_BIT,
            ScalarQuantizationType.EIGHT_BIT };
        assertArrayEquals(expectedValues, ScalarQuantizationType.values());
    }

    public void testSQTypesValueOf() {
        assertEquals(ScalarQuantizationType.ONE_BIT, ScalarQuantizationType.valueOf("ONE_BIT"));
        assertEquals(ScalarQuantizationType.TWO_BIT, ScalarQuantizationType.valueOf("TWO_BIT"));
        assertEquals(ScalarQuantizationType.FOUR_BIT, ScalarQuantizationType.valueOf("FOUR_BIT"));
        assertEquals(ScalarQuantizationType.EIGHT_BIT, ScalarQuantizationType.valueOf("EIGHT_BIT"));
    }

    public void testUniqueSQTypeValues() {
        Set<Integer> uniqueIds = new HashSet<>();
        for (ScalarQuantizationType type : ScalarQuantizationType.values()) {
            boolean added = uniqueIds.add(type.getId());
            assertTrue("Duplicate value found: " + type.getId(), added);
        }
    }
}
