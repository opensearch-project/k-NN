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

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.util.InfoStream;
import org.junit.Assert;
import org.mockito.Mockito;
import org.opensearch.knn.index.codec.KNNCodecTestCase;

public class NativeEngineFieldVectorsWriterTests extends KNNCodecTestCase {

    @SuppressWarnings("unchecked")
    @SneakyThrows
    public void testCreate_ForDifferentInputs_thenSuccess() {
        final FieldInfo fieldInfo = Mockito.mock(FieldInfo.class);
        Mockito.when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.FLOAT32);
        final FlatFieldVectorsWriter<float[]> mockedFlatFieldVectorsWriter = Mockito.mock(FlatFieldVectorsWriter.class);
        NativeEngineFieldVectorsWriter<float[]> floatWriter = (NativeEngineFieldVectorsWriter<float[]>) NativeEngineFieldVectorsWriter
            .create(fieldInfo, mockedFlatFieldVectorsWriter, InfoStream.getDefault());
        final float[] floatVector = new float[] { 1.0f, 2.0f };
        floatWriter.addValue(1, floatVector);
        Mockito.doNothing().when(mockedFlatFieldVectorsWriter).addValue(1, floatVector);

        Mockito.verify(fieldInfo).getVectorEncoding();
        Mockito.verify(mockedFlatFieldVectorsWriter).addValue(1, floatVector);

        final byte[] byteVector = new byte[] { 1, 2 };
        final FlatFieldVectorsWriter<byte[]> mockedFlatFieldByteVectorsWriter = Mockito.mock(FlatFieldVectorsWriter.class);
        Mockito.doNothing().when(mockedFlatFieldByteVectorsWriter).addValue(1, byteVector);
        Mockito.when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.BYTE);
        NativeEngineFieldVectorsWriter<byte[]> byteWriter = (NativeEngineFieldVectorsWriter<byte[]>) NativeEngineFieldVectorsWriter.create(
            fieldInfo,
            mockedFlatFieldByteVectorsWriter,
            InfoStream.getDefault()
        );
        Assert.assertNotNull(byteWriter);
        Mockito.verify(fieldInfo, Mockito.times(2)).getVectorEncoding();
        byteWriter.addValue(1, byteVector);
        Mockito.verify(mockedFlatFieldByteVectorsWriter).addValue(1, byteVector);
    }

    @SuppressWarnings("unchecked")
    @SneakyThrows
    public void testAddValue_ForDifferentInputs_thenSuccess() {
        final FieldInfo fieldInfo = Mockito.mock(FieldInfo.class);
        Mockito.when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.FLOAT32);
        final FlatFieldVectorsWriter<float[]> mockedFlatFieldVectorsWriter = Mockito.mock(FlatFieldVectorsWriter.class);
        final float[] vec1 = new float[] { 1.0f, 2.0f };
        final float[] vec2 = new float[] { 2.0f, 2.0f };
        Mockito.doNothing().when(mockedFlatFieldVectorsWriter).addValue(1, vec1);
        Mockito.doNothing().when(mockedFlatFieldVectorsWriter).addValue(2, vec2);
        final NativeEngineFieldVectorsWriter<float[]> floatWriter = (NativeEngineFieldVectorsWriter<float[]>) NativeEngineFieldVectorsWriter
            .create(fieldInfo, mockedFlatFieldVectorsWriter, InfoStream.getDefault());
        floatWriter.addValue(1, vec1);
        floatWriter.addValue(2, vec2);
        Mockito.verify(mockedFlatFieldVectorsWriter).addValue(1, vec1);
        Mockito.verify(mockedFlatFieldVectorsWriter).addValue(2, vec2);

        Assert.assertEquals(vec1, floatWriter.getVectors().get(1));
        Assert.assertEquals(vec2, floatWriter.getVectors().get(2));
        Mockito.verify(fieldInfo).getVectorEncoding();

        Mockito.when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.BYTE);
        final FlatFieldVectorsWriter<byte[]> mockedFlatFieldByteVectorsWriter = Mockito.mock(FlatFieldVectorsWriter.class);
        final byte[] bvec1 = new byte[] { 1, 2 };
        final byte[] bvec2 = new byte[] { 2, 2 };
        Mockito.doNothing().when(mockedFlatFieldByteVectorsWriter).addValue(1, bvec1);
        Mockito.doNothing().when(mockedFlatFieldByteVectorsWriter).addValue(2, bvec2);
        final NativeEngineFieldVectorsWriter<byte[]> byteWriter = (NativeEngineFieldVectorsWriter<byte[]>) NativeEngineFieldVectorsWriter
            .create(fieldInfo, mockedFlatFieldByteVectorsWriter, InfoStream.getDefault());
        byteWriter.addValue(1, bvec1);
        byteWriter.addValue(2, bvec2);

        Assert.assertEquals(bvec1, byteWriter.getVectors().get(1));
        Assert.assertEquals(bvec2, byteWriter.getVectors().get(2));
        Mockito.verify(fieldInfo, Mockito.times(2)).getVectorEncoding();
        Mockito.verify(mockedFlatFieldByteVectorsWriter).addValue(1, bvec1);
        Mockito.verify(mockedFlatFieldByteVectorsWriter).addValue(2, bvec2);
    }

    @SuppressWarnings("unchecked")
    @SneakyThrows
    public void testCopyValue_whenValidInput_thenException() {
        final FieldInfo fieldInfo = Mockito.mock(FieldInfo.class);
        FlatFieldVectorsWriter<?> mockedFlatFieldVectorsWriter = Mockito.mock(FlatFieldVectorsWriter.class);
        Mockito.when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.FLOAT32);
        final NativeEngineFieldVectorsWriter<float[]> floatWriter = (NativeEngineFieldVectorsWriter<float[]>) NativeEngineFieldVectorsWriter
            .create(fieldInfo, mockedFlatFieldVectorsWriter, InfoStream.getDefault());
        expectThrows(UnsupportedOperationException.class, () -> floatWriter.copyValue(new float[3]));

        Mockito.when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.BYTE);
        final NativeEngineFieldVectorsWriter<byte[]> byteWriter = (NativeEngineFieldVectorsWriter<byte[]>) NativeEngineFieldVectorsWriter
            .create(fieldInfo, mockedFlatFieldVectorsWriter, InfoStream.getDefault());
        expectThrows(UnsupportedOperationException.class, () -> byteWriter.copyValue(new byte[3]));
    }

    @SuppressWarnings("unchecked")
    @SneakyThrows
    public void testRamByteUsed_whenValidInput_thenSuccess() {
        final FieldInfo fieldInfo = Mockito.mock(FieldInfo.class);
        Mockito.when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.FLOAT32);
        Mockito.when(fieldInfo.getVectorDimension()).thenReturn(2);
        FlatFieldVectorsWriter<?> mockedFlatFieldVectorsWriter = Mockito.mock(FlatFieldVectorsWriter.class);
        Mockito.when(mockedFlatFieldVectorsWriter.ramBytesUsed()).thenReturn(1L);
        Mockito.when(mockedFlatFieldVectorsWriter.getDocsWithFieldSet()).thenReturn(new DocsWithFieldSet());
        final NativeEngineFieldVectorsWriter<float[]> floatWriter = (NativeEngineFieldVectorsWriter<float[]>) NativeEngineFieldVectorsWriter
            .create(fieldInfo, mockedFlatFieldVectorsWriter, InfoStream.getDefault());
        // testing for value > 0 as we don't have a concrete way to find out expected bytes. This can OS dependent too.
        Assert.assertTrue(floatWriter.ramBytesUsed() > 0);

        Mockito.when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.BYTE);
        final NativeEngineFieldVectorsWriter<byte[]> byteWriter = (NativeEngineFieldVectorsWriter<byte[]>) NativeEngineFieldVectorsWriter
            .create(fieldInfo, mockedFlatFieldVectorsWriter, InfoStream.getDefault());
        // testing for value > 0 as we don't have a concrete way to find out expected bytes. This can OS dependent too.
        Assert.assertTrue(byteWriter.ramBytesUsed() > 0);
        Mockito.verify(mockedFlatFieldVectorsWriter, Mockito.times(2)).ramBytesUsed();
    }
}
