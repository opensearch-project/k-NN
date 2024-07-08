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

public class SQTypesTests extends KNNTestCase {
    public void testSQTypesValues() {
        SQTypes[] expectedValues = {
                SQTypes.FP16,
                SQTypes.INT8,
                SQTypes.INT6,
                SQTypes.INT4,
                SQTypes.ONE_BIT,
                SQTypes.TWO_BIT
        };
        assertArrayEquals(expectedValues, SQTypes.values());
    }

    public void testSQTypesValueOf() {
        assertEquals(SQTypes.FP16, SQTypes.valueOf("FP16"));
        assertEquals(SQTypes.INT8, SQTypes.valueOf("INT8"));
        assertEquals(SQTypes.INT6, SQTypes.valueOf("INT6"));
        assertEquals(SQTypes.INT4, SQTypes.valueOf("INT4"));
        assertEquals(SQTypes.ONE_BIT, SQTypes.valueOf("ONE_BIT"));
        assertEquals(SQTypes.TWO_BIT, SQTypes.valueOf("TWO_BIT"));
    }
}
