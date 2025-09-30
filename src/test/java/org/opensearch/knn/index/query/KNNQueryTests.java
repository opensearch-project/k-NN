/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.opensearch.knn.index.VectorDataType;
import org.opensearch.test.OpenSearchTestCase;

public class KNNQueryTests extends OpenSearchTestCase {

    public void getQueryDimensions() {
        KNNQuery query = KNNQuery.builder().queryVector(new float[] { 1.0f, 2.0f, 3.0f }).vectorDataType(VectorDataType.FLOAT).build();
        assertEquals(3, query.getQueryDimension());
        query = KNNQuery.builder().queryVector(new float[] { 1.0f, 2.0f, 3.0f }).vectorDataType(VectorDataType.BYTE).build();
        assertEquals(3, query.getQueryDimension());
        query = KNNQuery.builder().byteQueryVector(new byte[] { 0, 1 }).vectorDataType(VectorDataType.BINARY).build();
        assertEquals(16, query.getQueryDimension());

        query = KNNQuery.builder().queryVector(new float[] { 0, 1 }).build();
        assertEquals(2, query.getQueryDimension());
    }

    public void testGetVectorFloat() {
        float[] floatVector = { 1.0f, 2.0f, 3.0f };
        KNNQuery query = KNNQuery.builder().queryVector(floatVector).vectorDataType(VectorDataType.FLOAT).build();
        assertTrue(query.getVector() instanceof float[]);
        assertEquals(floatVector, query.getVector());
    }

    public void testGetVectorBinary() {
        byte[] binaryVector = { 0, 1, 2 };
        KNNQuery query = KNNQuery.builder().byteQueryVector(binaryVector).vectorDataType(VectorDataType.BINARY).build();
        assertTrue(query.getVector() instanceof byte[]);
        assertEquals(binaryVector, query.getVector());
    }

    public void testGetVectorByteMemoryOptimized() {
        byte[] byteVector = { 1, 2, 3 };
        KNNQuery query = KNNQuery.builder()
            .byteQueryVector(byteVector)
            .vectorDataType(VectorDataType.BYTE)
            .isMemoryOptimizedSearch(true)
            .build();
        assertTrue(query.getVector() instanceof byte[]);
        assertEquals(byteVector, query.getVector());
    }

    public void testGetVectorByteNotMemoryOptimized() {
        float[] floatVector = { 1.0f, 2.0f, 3.0f };
        KNNQuery query = KNNQuery.builder()
            .queryVector(floatVector)
            .vectorDataType(VectorDataType.BYTE)
            .isMemoryOptimizedSearch(false)
            .build();
        assertTrue(query.getVector() instanceof float[]);
        assertEquals(floatVector, query.getVector());
    }
}
