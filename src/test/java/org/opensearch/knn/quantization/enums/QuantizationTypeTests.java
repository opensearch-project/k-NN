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

public class QuantizationTypeTests extends KNNTestCase {

    public void testQuantizationTypeValues() {
        QuantizationType[] expectedValues = {
                QuantizationType.SPACE_QUANTIZATION,
                QuantizationType.VALUE_QUANTIZATION
        };
        assertArrayEquals(expectedValues, QuantizationType.values());
    }

    public void testQuantizationTypeValueOf() {
        assertEquals(QuantizationType.SPACE_QUANTIZATION, QuantizationType.valueOf("SPACE_QUANTIZATION"));
        assertEquals(QuantizationType.VALUE_QUANTIZATION, QuantizationType.valueOf("VALUE_QUANTIZATION"));
    }
}
