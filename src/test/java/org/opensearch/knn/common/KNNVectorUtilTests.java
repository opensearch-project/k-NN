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

import lombok.SneakyThrows;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;

import java.util.List;

import static org.opensearch.knn.common.KNNVectorUtil.iterateVectorValuesOnce;

public class KNNVectorUtilTests extends KNNTestCase {
    public void testByteZeroVector() {
        assertTrue(KNNVectorUtil.isZeroVector(new byte[] { 0, 0, 0 }));
        assertFalse(KNNVectorUtil.isZeroVector(new byte[] { 1, 1, 1 }));
    }

    public void testFloatZeroVector() {
        assertTrue(KNNVectorUtil.isZeroVector(new float[] { 0.0f, 0.0f, 0.0f }));
        assertFalse(KNNVectorUtil.isZeroVector(new float[] { 1.0f, 1.0f, 1.0f }));
    }

    public void testIntListToArray() {
        assertArrayEquals(new int[] { 1, 2, 3 }, KNNVectorUtil.intListToArray(List.of(1, 2, 3)));
        assertNull(KNNVectorUtil.intListToArray(List.of()));
        assertNull(KNNVectorUtil.intListToArray(null));
    }

    @SneakyThrows
    public void testInit() {
        // Give
        final List<float[]> floatArray = List.of(new float[] { 1, 2 }, new float[] { 2, 3 });
        final int dimension = floatArray.get(0).length;
        final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
            floatArray
        );
        final KNNVectorValues<float[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, randomVectorValues);

        // When
        iterateVectorValuesOnce(knnVectorValues);

        // Then
        assertNotEquals(-1, knnVectorValues.docId());
        assertArrayEquals(floatArray.get(0), knnVectorValues.getVector(), 0.001f);
        assertEquals(dimension, knnVectorValues.dimension());
    }
}
