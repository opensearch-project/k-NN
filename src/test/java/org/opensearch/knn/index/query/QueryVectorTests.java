/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.junit.Test;
import static org.junit.Assert.*;

public class QueryVectorTests {

    @Test
    public void testFloatVectorConstructor() {
        float[] floatVector = new float[] { 1.0f, 2.0f, 3.0f };
        QueryVector queryVector = new QueryVector(floatVector);

        assertArrayEquals(floatVector, queryVector.getFloatVector(), 0.0001f);
        assertNull(queryVector.getByteVector());
    }

    @Test
    public void testByteVectorConstructor() {
        byte[] byteVector = new byte[] { 1, 2, 3 };
        QueryVector queryVector = new QueryVector(byteVector);

        assertArrayEquals(byteVector, queryVector.getByteVector());
        assertNull(queryVector.getFloatVector());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testBothVectorsConstructor_ThrowsException() {
        float[] floatVector = new float[] { 1.0f, 2.0f, 3.0f };
        byte[] byteVector = new byte[] { 1, 2, 3 };

        new QueryVector(floatVector, byteVector);
    }

    @Test
    public void testNullVectorsConstructor() {
        QueryVector queryVector = new QueryVector(null, null);

        assertNull(queryVector.getFloatVector());
        assertNull(queryVector.getByteVector());
    }

    @Test
    public void testEmptyVectors() {
        float[] emptyFloat = new float[0];
        byte[] emptyByte = new byte[0];

        QueryVector floatVector = new QueryVector(emptyFloat);
        QueryVector byteVector = new QueryVector(emptyByte);

        assertArrayEquals(emptyFloat, floatVector.getFloatVector(), 0.0001f);
        assertArrayEquals(emptyByte, byteVector.getByteVector());
    }
}
