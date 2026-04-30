/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.vectorvalues;

import lombok.SneakyThrows;
import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.store.ByteBuffersIndexOutput;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.packed.DirectMonotonicReader;
import org.opensearch.common.lucene.store.ByteArrayIndexInput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;
import org.opensearch.knn.memoryoptsearch.faiss.MonotonicIntegerSequenceEncoder;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;

public class FaissFloatVectorValuesTests extends KNNTestCase {

    private static final int DIMENSION = 4;
    private static final int NUM_VECTORS = 5;
    // Sparse mapping: internalId 0->0, 1->2, 2->5, 3->7, 4->10
    private static final int[] DOC_ID_MAPPING = { 0, 2, 5, 7, 10 };

    @SneakyThrows
    public void testSparseVectorsValues_whenValidInput_thenSuccess() {
        final FaissFloatVectorValues baseValues = createFaissFloatVectorValues();
        final DirectMonotonicReader idReader = createIdMappingReader(DOC_ID_MAPPING);
        final FaissFloatVectorValues.SparseFloatVectorValuesImpl sparse = new FaissFloatVectorValues.SparseFloatVectorValuesImpl(
            baseValues,
            idReader
        );

        // Verify dimension and size delegation
        assertEquals(DIMENSION, sparse.dimension());
        assertEquals(NUM_VECTORS, sparse.size());
        assertEquals(DIMENSION * Float.BYTES, sparse.getVectorByteLength());

        // Verify vectorValue delegation
        for (int i = 0; i < NUM_VECTORS; i++) {
            float[] expected = baseValues.vectorValue(i);
            float[] actual = sparse.vectorValue(i);
            assertArrayEquals(expected, actual, 0f);
        }

        // Verify ordToDoc mapping
        for (int i = 0; i < NUM_VECTORS; i++) {
            assertEquals(DOC_ID_MAPPING[i], sparse.ordToDoc(i));
        }

        // Verify getSlice returns the underlying IndexInput
        assertNotNull(sparse.getSlice());
        assertEquals(baseValues.getSlice(), sparse.getSlice());
    }

    @SneakyThrows
    public void testSparseVectorsValues_getAcceptOrds_whenAcceptDocsProvided_thenFiltersCorrectly() {
        final FaissFloatVectorValues baseValues = createFaissFloatVectorValues();
        final DirectMonotonicReader idReader = createIdMappingReader(DOC_ID_MAPPING);
        final FaissFloatVectorValues.SparseFloatVectorValuesImpl sparse = new FaissFloatVectorValues.SparseFloatVectorValuesImpl(
            baseValues,
            idReader
        );

        // acceptDocs accepts only even doc IDs
        Bits acceptDocs = new Bits() {
            @Override
            public boolean get(int index) {
                return index % 2 == 0;
            }

            @Override
            public int length() {
                return 11;
            }
        };

        Bits acceptOrds = sparse.getAcceptOrds(acceptDocs);
        assertNotNull(acceptOrds);
        assertEquals(NUM_VECTORS, acceptOrds.length());

        // internalId 0 -> docId 0 (even) -> true
        assertTrue(acceptOrds.get(0));
        // internalId 1 -> docId 2 (even) -> true
        assertTrue(acceptOrds.get(1));
        // internalId 2 -> docId 5 (odd) -> false
        assertFalse(acceptOrds.get(2));
        // internalId 3 -> docId 7 (odd) -> false
        assertFalse(acceptOrds.get(3));
        // internalId 4 -> docId 10 (even) -> true
        assertTrue(acceptOrds.get(4));
    }

    @SneakyThrows
    public void testSparseVectorsValues_getAcceptOrds_whenNull_thenReturnsNull() {
        final FaissFloatVectorValues baseValues = createFaissFloatVectorValues();
        final DirectMonotonicReader idReader = createIdMappingReader(DOC_ID_MAPPING);
        final FaissFloatVectorValues.SparseFloatVectorValuesImpl sparse = new FaissFloatVectorValues.SparseFloatVectorValuesImpl(
            baseValues,
            idReader
        );

        assertNull(sparse.getAcceptOrds(null));
    }

    @SneakyThrows
    public void testSparseVectorsValues_copy_thenReturnsCopy() {
        final FaissFloatVectorValues baseValues = createFaissFloatVectorValues();
        final DirectMonotonicReader idReader = createIdMappingReader(DOC_ID_MAPPING);
        final FaissFloatVectorValues.SparseFloatVectorValuesImpl sparse = new FaissFloatVectorValues.SparseFloatVectorValuesImpl(
            baseValues,
            idReader
        );

        var copy = sparse.copy();
        assertNotNull(copy);
        assertTrue(copy instanceof FaissFloatVectorValues.SparseFloatVectorValuesImpl);
        assertEquals(sparse.dimension(), copy.dimension());
        assertEquals(sparse.size(), copy.size());
    }

    public void testSparseVectorsValues_whenNonHasIndexSlice_thenThrows() {
        final List<float[]> vectors = List.of(new float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        final TestVectorValues.PreDefinedFloatVectorValues nonSliceValues = new TestVectorValues.PreDefinedFloatVectorValues(vectors);

        expectThrows(IllegalArgumentException.class, () -> new FaissFloatVectorValues.SparseFloatVectorValuesImpl(nonSliceValues, null));
    }

    private FaissFloatVectorValues createFaissFloatVectorValues() {
        int oneVectorByteSize = DIMENSION * Float.BYTES;
        ByteBuffer buf = ByteBuffer.allocate(NUM_VECTORS * oneVectorByteSize).order(ByteOrder.BIG_ENDIAN);
        for (int i = 0; i < NUM_VECTORS; i++) {
            for (int j = 0; j < DIMENSION; j++) {
                buf.putFloat(i * 10.0f + j);
            }
        }
        IndexInput indexInput = new ByteArrayIndexInput("test", buf.array());
        return new FaissFloatVectorValues(indexInput, oneVectorByteSize, DIMENSION, NUM_VECTORS);
    }

    @SneakyThrows
    private DirectMonotonicReader createIdMappingReader(int[] mapping) {
        ByteBuffersDataOutput dataOutput = new ByteBuffersDataOutput();
        ByteBuffersIndexOutput indexOutput = new ByteBuffersIndexOutput(dataOutput, "test", "test");
        for (int docId : mapping) {
            indexOutput.writeLong(docId);
        }
        indexOutput.close();
        IndexInput input = new ByteArrayIndexInput("test", dataOutput.toArrayCopy());
        return MonotonicIntegerSequenceEncoder.encode(mapping.length, input);
    }
}
