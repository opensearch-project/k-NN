/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.mockito.Mock;
import org.opensearch.knn.KNNTestCase;

import java.util.List;
import java.util.Map;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class KNNMergeVectorValuesTests extends KNNTestCase {

    private static final String FIELD = "field";

    @Mock
    private KnnVectorsReader knnVectorsReader1;
    @Mock
    private KnnVectorsReader knnVectorsReader2;
    @Mock
    private KnnVectorsReader knnVectorsReader3;
    @Mock
    private FixedBitSet fixedBitSetLiveDocs;
    @Mock
    private Bits fixedBitsLiveDocs;

    @SneakyThrows
    public void testFloatMergeVectorValues() {
        // Given
        final KnnVectorsReader[] knnVectorsReaders = { knnVectorsReader1, knnVectorsReader2, knnVectorsReader3 };
        final Bits[] liveDocs = { fixedBitSetLiveDocs, fixedBitsLiveDocs, null };
        final Map<Integer, float[]> floats1 = Map.of(0, new float[] { 1, 2 }, 1, new float[] { 2, 3 });
        final List<float[]> floats1List = List.of(new float[] { 1, 2 }, new float[] { 2, 3 });
        final Map<Integer, float[]> floats2 = Map.of(0, new float[] { 3, 4 }, 1, new float[] { 4, 6 });
        final List<float[]> floats2List = List.of(new float[] { 3, 4 }, new float[] { 4, 6 });
        final Map<Integer, float[]> floats3 = Map.of(0, new float[] { 1, 2 });
        final List<float[]> floats3List = List.of(new float[] { 1, 2 });
        final MergeState.DocMap[] docMaps = {
            (docId) -> floats1.get(docId) != null ? docId : -1,
            (docId) -> floats2.get(docId) != null ? docId + 2 : -1,
            (docId) -> floats3.get(docId) != null ? docId + 4 : -1 };

        Map<Integer, float[]> floats = Map.of(
            0,
            new float[] { 1, 2 },
            1,
            new float[] { 2, 3 },
            2,
            new float[] { 3, 4 },
            3,
            new float[] { 4, 6 },
            4,
            new float[] { 1, 2 }
        );
        final TestVectorValues.PreDefinedFloatVectorValues vectorValues1 = new TestVectorValues.PreDefinedFloatVectorValues(floats1List);
        final TestVectorValues.PreDefinedFloatVectorValues vectorValues2 = new TestVectorValues.PreDefinedFloatVectorValues(floats2List);
        final TestVectorValues.PreDefinedFloatVectorValues vectorValues3 = new TestVectorValues.PreDefinedFloatVectorValues(floats3List);

        FieldInfo fieldInfo = fieldInfo(0, VectorEncoding.FLOAT32);
        when(knnVectorsReader1.getFloatVectorValues(FIELD)).thenReturn(vectorValues1);
        when(knnVectorsReader2.getFloatVectorValues(FIELD)).thenReturn(vectorValues2);
        when(knnVectorsReader3.getFloatVectorValues(FIELD)).thenReturn(vectorValues3);
        when(fixedBitSetLiveDocs.cardinality()).thenReturn(2);
        when(fixedBitsLiveDocs.length()).thenReturn(3);
        when(fixedBitsLiveDocs.get(0)).thenReturn(true);
        when(fixedBitsLiveDocs.get(1)).thenReturn(false);
        when(fixedBitsLiveDocs.get(2)).thenReturn(true);

        final MergeState mergeState = mergeState(knnVectorsReaders, liveDocs, docMaps);

        // When
        KNNVectorValuesIterator.MergeFloat32VectorValuesIterator iterator = KNNMergeVectorValues.mergeFloatVectorValues(
            fieldInfo,
            mergeState
        );

        // Then
        assertEquals(5, iterator.liveDocs());
        int size = 0;
        while (iterator.nextDoc() != NO_MORE_DOCS) {
            assertArrayEquals(floats.get(iterator.docId()), iterator.vectorValue(), 0.01f);
            size++;
        }
        assertEquals(floats.size(), size);
    }

    @SneakyThrows
    public void testByteMergeVectorValues() {
        // Given
        final KnnVectorsReader[] knnVectorsReaders = { knnVectorsReader1, knnVectorsReader2, knnVectorsReader3 };
        final Bits[] liveDocs = { fixedBitSetLiveDocs, fixedBitsLiveDocs, null };
        final Map<Integer, byte[]> floats1 = Map.of(0, new byte[] { 1, 2 }, 1, new byte[] { 2, 3 });
        final List<byte[]> floats1List = List.of(new byte[] { 1, 2 }, new byte[] { 2, 3 });
        final Map<Integer, byte[]> floats2 = Map.of(0, new byte[] { 3, 4 }, 1, new byte[] { 4, 6 });
        final List<byte[]> floats2List = List.of(new byte[] { 3, 4 }, new byte[] { 4, 6 });
        final Map<Integer, byte[]> floats3 = Map.of(0, new byte[] { 1, 2 });
        final List<byte[]> floats3List = List.of(new byte[] { 1, 2 });
        final MergeState.DocMap[] docMaps = {
            (docId) -> floats1.get(docId) != null ? docId : -1,
            (docId) -> floats2.get(docId) != null ? docId + 2 : -1,
            (docId) -> floats3.get(docId) != null ? docId + 4 : -1 };

        Map<Integer, byte[]> floats = Map.of(
            0,
            new byte[] { 1, 2 },
            1,
            new byte[] { 2, 3 },
            2,
            new byte[] { 3, 4 },
            3,
            new byte[] { 4, 6 },
            4,
            new byte[] { 1, 2 }
        );
        final TestVectorValues.PreDefinedByteVectorValues vectorValues1 = new TestVectorValues.PreDefinedByteVectorValues(floats1List);
        final TestVectorValues.PreDefinedByteVectorValues vectorValues2 = new TestVectorValues.PreDefinedByteVectorValues(floats2List);
        final TestVectorValues.PreDefinedByteVectorValues vectorValues3 = new TestVectorValues.PreDefinedByteVectorValues(floats3List);

        FieldInfo fieldInfo = fieldInfo(0, VectorEncoding.BYTE);
        when(knnVectorsReader1.getByteVectorValues(FIELD)).thenReturn(vectorValues1);
        when(knnVectorsReader2.getByteVectorValues(FIELD)).thenReturn(vectorValues2);
        when(knnVectorsReader3.getByteVectorValues(FIELD)).thenReturn(vectorValues3);
        when(fixedBitSetLiveDocs.cardinality()).thenReturn(2);
        when(fixedBitsLiveDocs.length()).thenReturn(3);
        when(fixedBitsLiveDocs.get(0)).thenReturn(true);
        when(fixedBitsLiveDocs.get(1)).thenReturn(false);
        when(fixedBitsLiveDocs.get(2)).thenReturn(true);

        final MergeState mergeState = mergeState(knnVectorsReaders, liveDocs, docMaps);

        // When
        KNNVectorValuesIterator.MergeByteVectorValuesIterator iterator = KNNMergeVectorValues.mergeByteVectorValues(fieldInfo, mergeState);

        // Then
        assertEquals(5, iterator.liveDocs());
        int size = 0;
        while (iterator.nextDoc() != NO_MORE_DOCS) {
            assertArrayEquals(floats.get(iterator.docId()), iterator.vectorValue());
            size++;
        }
        assertEquals(floats.size(), size);
    }

    private MergeState mergeState(KnnVectorsReader[] knnVectorsReaders, Bits[] liveDocs, MergeState.DocMap[] docMaps) {
        return new MergeState(
            docMaps,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            liveDocs,
            null,
            null,
            knnVectorsReaders,
            null,
            null,
            null,
            false
        );
    }

    private FieldInfo fieldInfo(int fieldNumber, VectorEncoding vectorEncoding) {
        FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.getFieldNumber()).thenReturn(fieldNumber);
        when(fieldInfo.getVectorEncoding()).thenReturn(vectorEncoding);
        when(fieldInfo.getName()).thenReturn(FIELD);
        when(fieldInfo.hasVectorValues()).thenReturn(true);
        return fieldInfo;
    }
}
