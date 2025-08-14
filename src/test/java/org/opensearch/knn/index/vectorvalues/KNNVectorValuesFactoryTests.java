/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import lombok.SneakyThrows;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.VectorEncoding;
import org.junit.Assert;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.VectorDataType;

import java.util.List;
import java.util.Map;

public class KNNVectorValuesFactoryTests extends KNNTestCase {
    private static final int COUNT = 10;
    private static final int DIMENSION = 10;

    public void testGetVectorValuesFromDISI_whenValidInput_thenSuccess() {
        final BinaryDocValues binaryDocValues = new TestVectorValues.RandomVectorBinaryDocValues(COUNT, DIMENSION);
        final KNNVectorValues<float[]> floatVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, binaryDocValues);
        Assert.assertNotNull(floatVectorValues);

        final KNNVectorValues<float[]> halfFloatVectorValues = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.HALF_FLOAT,
            binaryDocValues
        );
        Assert.assertNotNull(halfFloatVectorValues);

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

        final KNNVectorValues<float[]> halfFloatVectorValues = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.HALF_FLOAT,
            docsWithFieldSet,
            floatVectorMap
        );
        Assert.assertNotNull(halfFloatVectorValues);

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
    public void testGetVectorValuesFromFieldInfo_whenVectorDimIsNotZero_thenSuccess() {
        final List<byte[]> byteArrayList = List.of(new byte[] { 1, 2, 3 });
        final List<float[]> floatArrayList = List.of(new float[] { 1.3f, 2.2f, 3.2f });
        final List<byte[]> binaryArrayList = List.of(new byte[] { 3, 2, 3 });
        final FieldInfo fieldInfo = Mockito.mock(FieldInfo.class);
        final SegmentReader reader = Mockito.mock(SegmentReader.class);
        Mockito.when(fieldInfo.hasVectorValues()).thenReturn(true);
        Mockito.when(fieldInfo.getName()).thenReturn("test_field");

        // Checking for ByteVectorValues
        Mockito.when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.BYTE);
        Mockito.when(fieldInfo.getAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD)).thenReturn(VectorDataType.BYTE.getValue());
        Mockito.when(reader.getByteVectorValues("test_field")).thenReturn(new TestVectorValues.PreDefinedByteVectorValues(byteArrayList));
        final KNNVectorValues<byte[]> byteVectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
        byteVectorValues.nextDoc();
        Assert.assertArrayEquals(byteArrayList.get(0), byteVectorValues.getVector());
        Assert.assertNotNull(byteVectorValues);

        // Checking for FloatVectorValues
        Mockito.when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.FLOAT32);
        Mockito.when(fieldInfo.getAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD)).thenReturn(VectorDataType.FLOAT.getValue());
        Mockito.when(reader.getFloatVectorValues("test_field"))
            .thenReturn(new TestVectorValues.PreDefinedFloatVectorValues(floatArrayList));
        final KNNVectorValues<float[]> floatVectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
        floatVectorValues.nextDoc();
        Assert.assertArrayEquals(floatArrayList.get(0), floatVectorValues.getVector(), 0.0f);
        Assert.assertNotNull(floatVectorValues);

        // Checking for HALF_FLOAT
        Mockito.when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.FLOAT32);
        Mockito.when(fieldInfo.getAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD)).thenReturn(VectorDataType.HALF_FLOAT.getValue());
        Mockito.when(reader.getFloatVectorValues("test_field"))
            .thenReturn(new TestVectorValues.PreDefinedFloatVectorValues(floatArrayList));
        final KNNVectorValues<float[]> halfFloatVectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
        halfFloatVectorValues.nextDoc();
        Assert.assertArrayEquals(floatArrayList.get(0), halfFloatVectorValues.getVector(), 0.0f);
        Assert.assertNotNull(halfFloatVectorValues);

        // Checking for BinaryVectorValues
        Mockito.when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.BYTE);
        Mockito.when(fieldInfo.getAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD)).thenReturn(VectorDataType.BINARY.getValue());
        Mockito.when(reader.getByteVectorValues("test_field"))
            .thenReturn(new TestVectorValues.PreDefinedBinaryVectorValues(binaryArrayList));
        final KNNVectorValues<byte[]> binaryVectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
        binaryVectorValues.nextDoc();
        Assert.assertArrayEquals(binaryArrayList.get(0), binaryVectorValues.getVector());
        Assert.assertNotNull(binaryVectorValues);

    }

    @SneakyThrows
    public void testGetVectorValuesFromFieldInfo_whenVectorDimIsZero_thenSuccess() {
        final List<byte[]> byteArrayList = List.of(new byte[] { 1, 2, 3 });
        final List<float[]> floatArrayList = List.of(new float[] { 1.3f, 2.2f, 3.2f });
        final List<byte[]> binaryArrayList = List.of(new byte[] { 3, 2, 3 });
        final FieldInfo fieldInfo = Mockito.mock(FieldInfo.class);
        final SegmentReader reader = Mockito.mock(SegmentReader.class);
        Mockito.when(fieldInfo.hasVectorValues()).thenReturn(false);
        Mockito.when(fieldInfo.getName()).thenReturn("test_field");

        // Checking for ByteVectorValues
        Mockito.when(fieldInfo.getAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD)).thenReturn(VectorDataType.BYTE.getValue());
        Mockito.when(reader.getBinaryDocValues("test_field"))
            .thenReturn(new TestVectorValues.PredefinedByteVectorBinaryDocValues(byteArrayList));

        final KNNVectorValues<byte[]> byteVectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
        byteVectorValues.nextDoc();
        Assert.assertArrayEquals(byteArrayList.get(0), byteVectorValues.getVector());
        Assert.assertNotNull(byteVectorValues);

        // Checking for Floats with BinaryDocValues
        Mockito.when(fieldInfo.getAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD)).thenReturn(VectorDataType.FLOAT.getValue());
        Mockito.when(reader.getBinaryDocValues("test_field"))
            .thenReturn(new TestVectorValues.PredefinedFloatVectorBinaryDocValues(floatArrayList));

        final KNNVectorValues<float[]> floatVectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
        floatVectorValues.nextDoc();
        Assert.assertArrayEquals(floatArrayList.get(0), floatVectorValues.getVector(), 0.0f);
        Assert.assertNotNull(floatVectorValues);

        // Checking for HALF_FLOAT with BinaryDocValues
        Mockito.when(fieldInfo.getAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD)).thenReturn(VectorDataType.HALF_FLOAT.getValue());
        Mockito.when(reader.getBinaryDocValues("test_field"))
            .thenReturn(new TestVectorValues.PredefinedFloatVectorBinaryDocValues(floatArrayList));

        final KNNVectorValues<float[]> halfFloatVectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
        halfFloatVectorValues.nextDoc();
        Assert.assertArrayEquals(floatArrayList.get(0), halfFloatVectorValues.getVector(), 0.0f);
        Assert.assertNotNull(halfFloatVectorValues);

        // Checking for BinaryVectorValues
        Mockito.when(fieldInfo.getAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD)).thenReturn(VectorDataType.BINARY.getValue());
        Mockito.when(reader.getBinaryDocValues("test_field"))
            .thenReturn(new TestVectorValues.PredefinedByteVectorBinaryDocValues(binaryArrayList));

        final KNNVectorValues<byte[]> binaryVectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
        binaryVectorValues.nextDoc();
        Assert.assertArrayEquals(binaryArrayList.get(0), binaryVectorValues.getVector());
        Assert.assertNotNull(binaryVectorValues);

        Mockito.verify(fieldInfo, Mockito.times(0)).getVectorEncoding();
    }

}
