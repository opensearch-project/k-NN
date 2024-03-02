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

package org.opensearch.knn.common;

import org.opensearch.knn.KNNTestCase;

public class KNNVectorUtilTests extends KNNTestCase {
    public void testByteZeroVector() {
        assertTrue(KNNVectorUtil.isZeroVector(new byte[] { 0, 0, 0 }));
        assertFalse(KNNVectorUtil.isZeroVector(new byte[] { 1, 1, 1 }));
    }

    public void testFloatZeroVector() {
        assertTrue(KNNVectorUtil.isZeroVector(new float[] { 0.0f, 0.0f, 0.0f }));
        assertFalse(KNNVectorUtil.isZeroVector(new float[] { 1.0f, 1.0f, 1.0f }));
    }
}
