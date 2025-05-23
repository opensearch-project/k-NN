/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.opensearch.knn.KNNTestCase;

import static org.apache.lucene.tests.util.LuceneTestCase.expectThrows;
import static org.junit.Assert.*;

public class QueryVectorTests extends KNNTestCase {

    public void testFloatVectorConstructor() {
        float[] floatVector = new float[] { 1.0f, 2.0f, 3.0f };
        QueryVector queryVector = new QueryVector(floatVector);

        assertArrayEquals(floatVector, queryVector.getFloatVector(), 0.0001f);
        assertNull(queryVector.getByteVector());
    }

    public void testByteVectorConstructor() {
        byte[] byteVector = new byte[] { 1, 2, 3 };
        QueryVector queryVector = new QueryVector(byteVector);

        assertArrayEquals(byteVector, queryVector.getByteVector());
        assertNull(queryVector.getFloatVector());
    }

    public void testBothVectorsConstructor_ThrowsException() {
        float[] floatVector = new float[] { 1.0f, 2.0f, 3.0f };
        byte[] byteVector = new byte[] { 1, 2, 3 };
        expectThrows(IllegalArgumentException.class, () -> { new QueryVector(floatVector, byteVector); });
    }

    public void testNullVectorsConstructor() {
        QueryVector queryVector = new QueryVector(null, null);

        assertNull(queryVector.getFloatVector());
        assertNull(queryVector.getByteVector());
    }

    public void testEmptyVectors() {
        float[] emptyFloat = new float[0];
        byte[] emptyByte = new byte[0];

        QueryVector floatVector = new QueryVector(emptyFloat);
        QueryVector byteVector = new QueryVector(emptyByte);

        assertArrayEquals(emptyFloat, floatVector.getFloatVector(), 0.0001f);
        assertArrayEquals(emptyByte, byteVector.getByteVector());
    }
}
