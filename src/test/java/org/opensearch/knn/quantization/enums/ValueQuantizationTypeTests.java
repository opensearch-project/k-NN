/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.quantization.enums;

import org.opensearch.knn.KNNTestCase;

public class ValueQuantizationTypeTests extends KNNTestCase {
    public void testValueQuantizationTypeValues() {
        ValueQuantizationType[] expectedValues = {
                ValueQuantizationType.SQ
        };
        assertArrayEquals(expectedValues, ValueQuantizationType.values());
    }

    public void testValueQuantizationTypeValueOf() {
        assertEquals(ValueQuantizationType.SQ, ValueQuantizationType.valueOf("SQ"));
    }
}
