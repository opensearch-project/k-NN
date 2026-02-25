/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.DocIdSetIterator;
import org.junit.Assert;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.VectorDataType;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

public class KNNVectorValuesFactoryTests extends KNNTestCase {
    private static final int COUNT = 10;
    private static final int DIMENSION = 10;

    public void testGetVectorValuesFromDISI_whenValidInput_thenSuccess() {
        final BinaryDocValues binaryDocValues = new TestVectorValues.RandomVectorBinaryDocValues(COUNT, DIMENSION);
        final KNNVectorValues<float[]> floatVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, binaryDocValues);
        Assert.assertNotNull(floatVectorValues);

        final KNNVectorValues<byte[]> byteVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.BYTE, binaryDocValues);
        Assert.assertNotNull(byteVectorValues);

        final KNNVectorValues<byte[]> binaryVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.BINARY, binaryDocValues);
        Assert.assertNotNull(binaryVectorValues);
    }

    public void testGetVectorValuesUsingDocWithFieldSet_whenValidInput_thenSuccess() {
        final DocsWithFieldSet docsWithFieldSet = new DocsWithFieldSet();
        docsWithFieldSet.add(0);
        docsWithFieldSet.add(1);
        final Map<Integer, float[]> floatVectorMap = Map.of(0, new float[] { 1, 2 }, 1, new float[] { 2, 3 });
        final KNNVectorValues<float[]> floatVectorValues = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.FLOAT,
            docsWithFieldSet,
            floatVectorMap
        );
        Assert.assertNotNull(floatVectorValues);

        final Map<Integer, byte[]> byteVectorMap = Map.of(0, new byte[] { 4, 5 }, 1, new byte[] { 6, 7 });

        final KNNVectorValues<byte[]> byteVectorValues = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.BYTE,
            docsWithFieldSet,
            byteVectorMap
        );
        Assert.assertNotNull(byteVectorValues);

        final KNNVectorValues<byte[]> binaryVectorValues = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.BINARY,
            docsWithFieldSet,
            byteVectorMap
        );
        Assert.assertNotNull(binaryVectorValues);
    }

    @SneakyThrows
    public void testGetVectorValuesFromFieldInfo_whenQueryVectorQuantized_thenSuccess() {
        final List<float[]> floatArrayList = List.of(
            new float[] { 1.3f, 2.2f, 3.2f },
            new float[] { 4.1f, 5.5f, 6.6f },
            new float[] { 7.7f, 8.8f, 9.9f },
            new float[] { 0.5f, 1.5f, 2.5f }
        );
        final List<byte[]> quantizedByteArrayList = List.of(
            new byte[] { 1, 0, 1 },
            new byte[] { 0, 1, 0 },
            new byte[] { 1, 1, 1 },
            new byte[] { 0, 0, 1 }
        );
        final FieldInfo fieldInfo = Mockito.mock(FieldInfo.class);
        final SegmentReader reader = Mockito.mock(SegmentReader.class);
        Mockito.when(fieldInfo.hasVectorValues()).thenReturn(true);
        Mockito.when(fieldInfo.getName()).thenReturn("test_field");
        Mockito.when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.FLOAT32);
        Mockito.when(fieldInfo.getAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD)).thenReturn(VectorDataType.FLOAT.getValue());
        Mockito.when(reader.getFloatVectorValues("test_field"))
            .thenReturn(new TestVectorValues.PreDefinedFloatVectorValues(floatArrayList));

        final KnnVectorsReader vectorsReader = Mockito.mock(KnnVectorsReader.class);
        Mockito.when(reader.getVectorReader()).thenReturn(vectorsReader);
        Mockito.when(vectorsReader.getByteVectorValues("test_field"))
            .thenReturn(new TestVectorValues.PreDefinedByteVectorValues(quantizedByteArrayList));

        final KNNVectorValues<byte[]> quantizedVectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader, true);
        Assert.assertNotNull(quantizedVectorValues);
        Assert.assertTrue(quantizedVectorValues instanceof KNNBinaryVectorValues);
        verify(reader).getFloatVectorValues("test_field");
        verify(vectorsReader).getByteVectorValues("test_field");
        assertByteVectorValues(quantizedVectorValues, quantizedByteArrayList);
    }

    @SneakyThrows
    public void testGetVectorValuesFromFieldInfo_whenQueryVectorNotQuantized_thenSuccess() {
        final List<float[]> floatArrayList = List.of(
            new float[] { 1.3f, 2.2f, 3.2f },
            new float[] { 4.1f, 5.5f, 6.6f },
            new float[] { 7.7f, 8.8f, 9.9f }
        );
        final FieldInfo fieldInfo = Mockito.mock(FieldInfo.class);
        final SegmentReader reader = Mockito.mock(SegmentReader.class);
        Mockito.when(fieldInfo.hasVectorValues()).thenReturn(true);
        Mockito.when(fieldInfo.getName()).thenReturn("test_field");
        Mockito.when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.FLOAT32);
        Mockito.when(fieldInfo.getAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD)).thenReturn(VectorDataType.FLOAT.getValue());
        Mockito.when(reader.getFloatVectorValues("test_field"))
            .thenReturn(new TestVectorValues.PreDefinedFloatVectorValues(floatArrayList));

        final KNNVectorValues<float[]> vectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader, false);
        Assert.assertNotNull(vectorValues);
        assertFloatVectorValues(vectorValues, floatArrayList);
        verify(reader).getFloatVectorValues("test_field");
        verify(reader, never()).getVectorReader();

    }

    @SneakyThrows
    public void testGetVectorValuesFromFieldInfo_whenVectorDimIsNotZero_thenSuccess() {
        final List<byte[]> byteArrayList = List.of(new byte[] { 1, 2, 3 }, new byte[] { 4, 5, 6 }, new byte[] { 7, 8, 9 });
        final List<float[]> floatArrayList = List.of(
            new float[] { 1.3f, 2.2f, 3.2f },
            new float[] { 4.1f, 5.5f, 6.6f },
            new float[] { 7.7f, 8.8f, 9.9f }
        );
        final List<byte[]> binaryArrayList = List.of(new byte[] { 1, 0, 1 }, new byte[] { 0, 1, 0 }, new byte[] { 1, 1, 1 });
        final FieldInfo fieldInfo = Mockito.mock(FieldInfo.class);
        final SegmentReader reader = Mockito.mock(SegmentReader.class);
        Mockito.when(fieldInfo.hasVectorValues()).thenReturn(true);
        Mockito.when(fieldInfo.getName()).thenReturn("test_field");

        // Checking for ByteVectorValues
        Mockito.when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.BYTE);
        Mockito.when(fieldInfo.getAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD)).thenReturn(VectorDataType.BYTE.getValue());
        Mockito.when(reader.getByteVectorValues("test_field")).thenReturn(new TestVectorValues.PreDefinedByteVectorValues(byteArrayList));
        final KNNVectorValues<byte[]> byteVectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
        Assert.assertNotNull(byteVectorValues);
        assertByteVectorValues(byteVectorValues, byteArrayList);

        // Checking for FloatVectorValues
        Mockito.when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.FLOAT32);
        Mockito.when(fieldInfo.getAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD)).thenReturn(VectorDataType.FLOAT.getValue());
        Mockito.when(reader.getFloatVectorValues("test_field"))
            .thenReturn(new TestVectorValues.PreDefinedFloatVectorValues(floatArrayList));
        final KNNVectorValues<float[]> floatVectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
        Assert.assertNotNull(floatVectorValues);
        assertFloatVectorValues(floatVectorValues, floatArrayList);

        // Checking for BinaryVectorValues
        Mockito.when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.BYTE);
        Mockito.when(fieldInfo.getAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD)).thenReturn(VectorDataType.BINARY.getValue());
        Mockito.when(reader.getByteVectorValues("test_field"))
            .thenReturn(new TestVectorValues.PreDefinedBinaryVectorValues(binaryArrayList));
        final KNNVectorValues<byte[]> binaryVectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
        Assert.assertNotNull(binaryVectorValues);
        assertByteVectorValues(binaryVectorValues, binaryArrayList);

    }

    @SneakyThrows
    public void testGetVectorValuesFromFieldInfo_whenVectorDimIsZero_thenSuccess() {
        final List<byte[]> byteArrayList = List.of(new byte[] { 1, 2, 3 }, new byte[] { 4, 5, 6 }, new byte[] { 7, 8, 9 });
        final List<float[]> floatArrayList = List.of(
            new float[] { 1.3f, 2.2f, 3.2f },
            new float[] { 4.1f, 5.5f, 6.6f },
            new float[] { 7.7f, 8.8f, 9.9f }
        );
        final List<byte[]> binaryArrayList = List.of(new byte[] { 1, 0, 1 }, new byte[] { 0, 1, 0 }, new byte[] { 1, 1, 1 });
        final FieldInfo fieldInfo = Mockito.mock(FieldInfo.class);
        final SegmentReader reader = Mockito.mock(SegmentReader.class);
        Mockito.when(fieldInfo.hasVectorValues()).thenReturn(false);
        Mockito.when(fieldInfo.getName()).thenReturn("test_field");

        // Checking for ByteVectorValues
        Mockito.when(fieldInfo.getAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD)).thenReturn(VectorDataType.BYTE.getValue());
        Mockito.when(reader.getBinaryDocValues("test_field"))
            .thenReturn(new TestVectorValues.PredefinedByteVectorBinaryDocValues(byteArrayList));

        final KNNVectorValues<byte[]> byteVectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
        Assert.assertNotNull(byteVectorValues);
        assertByteVectorValues(byteVectorValues, byteArrayList);

        // Checking for Floats with BinaryDocValues
        Mockito.when(fieldInfo.getAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD)).thenReturn(VectorDataType.FLOAT.getValue());
        Mockito.when(reader.getBinaryDocValues("test_field"))
            .thenReturn(new TestVectorValues.PredefinedFloatVectorBinaryDocValues(floatArrayList));

        final KNNVectorValues<float[]> floatVectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
        Assert.assertNotNull(floatVectorValues);
        assertFloatVectorValues(floatVectorValues, floatArrayList);

        // Checking for BinaryVectorValues
        Mockito.when(fieldInfo.getAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD)).thenReturn(VectorDataType.BINARY.getValue());
        Mockito.when(reader.getBinaryDocValues("test_field"))
            .thenReturn(new TestVectorValues.PredefinedByteVectorBinaryDocValues(binaryArrayList));

        final KNNVectorValues<byte[]> binaryVectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
        Assert.assertNotNull(binaryVectorValues);
        assertByteVectorValues(binaryVectorValues, binaryArrayList);

        verify(fieldInfo, Mockito.times(0)).getVectorEncoding();
    }

    private void assertByteVectorValues(KNNVectorValues<byte[]> vectorValues, List<byte[]> expectedVectors) throws IOException {
        for (byte[] expectedVector : expectedVectors) {
            vectorValues.nextDoc();
            Assert.assertArrayEquals(expectedVector, vectorValues.getVector());
        }
        assert (vectorValues.nextDoc() == DocIdSetIterator.NO_MORE_DOCS);
    }

    private void assertFloatVectorValues(KNNVectorValues<float[]> vectorValues, List<float[]> expectedVectors) throws IOException {
        for (float[] expectedVector : expectedVectors) {
            vectorValues.nextDoc();
            Assert.assertArrayEquals(expectedVector, vectorValues.getVector(), 0.0f);
        }
        assert (vectorValues.nextDoc() == DocIdSetIterator.NO_MORE_DOCS);
    }

}
