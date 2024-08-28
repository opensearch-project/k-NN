/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.search.Sort;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Version;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.quantization.models.quantizationState.MultiBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationStateReadConfig;

import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.times;

public class KNNQuantizationStateReaderTests extends KNNTestCase {

    @SneakyThrows
    public void testReadFromSegmentReadState() {
        final String segmentName = "test-segment-name";
        final String segmentSuffix = "test-segment-suffix";

        final SegmentInfo segmentInfo = new SegmentInfo(
            Mockito.mock(Directory.class),
            Mockito.mock(Version.class),
            Mockito.mock(Version.class),
            segmentName,
            0,
            false,
            false,
            Mockito.mock(Codec.class),
            Mockito.mock(Map.class),
            new byte[16],
            Mockito.mock(Map.class),
            Mockito.mock(Sort.class)
        );

        Directory directory = Mockito.mock(Directory.class);
        IndexInput input = Mockito.mock(IndexInput.class);
        Mockito.when(directory.openInput(any(), any())).thenReturn(input);

        String fieldName = "test-field";
        FieldInfos fieldInfos = Mockito.mock(FieldInfos.class);
        FieldInfo fieldInfo = Mockito.mock(FieldInfo.class);
        Mockito.when(fieldInfo.getName()).thenReturn(fieldName);
        Mockito.when(fieldInfos.fieldInfo(anyInt())).thenReturn(fieldInfo);

        final SegmentReadState segmentReadState = new SegmentReadState(
            directory,
            segmentInfo,
            fieldInfos,
            Mockito.mock(IOContext.class),
            segmentSuffix
        );

        try (MockedStatic<KNNQuantizationStateReader> mockedStaticReader = Mockito.mockStatic(KNNQuantizationStateReader.class)) {
            mockedStaticReader.when(() -> KNNQuantizationStateReader.getNumFields(input)).thenReturn(2);
            mockedStaticReader.when(() -> KNNQuantizationStateReader.read(segmentReadState)).thenCallRealMethod();
            try (MockedStatic<CodecUtil> mockedStaticCodecUtil = mockStatic(CodecUtil.class)) {
                KNNQuantizationStateReader.read(segmentReadState);

                mockedStaticCodecUtil.verify(() -> CodecUtil.retrieveChecksum(input));
                Mockito.verify(input, times(4)).readInt();
                Mockito.verify(input, times(2)).readVLong();
                Mockito.verify(input, times(2)).readBytes(any(byte[].class), anyInt(), anyInt());
                Mockito.verify(input, times(2)).seek(anyLong());
            }
        }
    }

    @SneakyThrows
    public void testReadFromQuantizationStateReadConfig() {
        Directory directory = Mockito.mock(Directory.class);
        IndexInput input = Mockito.mock(IndexInput.class);
        Mockito.when(directory.openInput(any(), any())).thenReturn(input);

        int fieldNumber = 4;
        FieldInfos fieldInfos = Mockito.mock(FieldInfos.class);
        FieldInfo fieldInfo = Mockito.mock(FieldInfo.class);
        Mockito.when(fieldInfo.getFieldNumber()).thenReturn(fieldNumber);
        Mockito.when(fieldInfos.fieldInfo(anyInt())).thenReturn(fieldInfo);

        String segmentName = "test-segment-name";
        String segmentSuffix = "test-segment-suffix";
        String scalarQuantizationTypeId1 = "1";
        String scalarQuantizationTypeId2 = "2";
        String scalarQuantizationTypeId4 = "4";
        String scalarQuantizationTypeIdIncorrect = "-1";
        QuantizationStateReadConfig quantizationStateReadConfig = Mockito.mock(QuantizationStateReadConfig.class);
        Mockito.when(quantizationStateReadConfig.getSegmentName()).thenReturn(segmentName);
        Mockito.when(quantizationStateReadConfig.getSegmentSuffix()).thenReturn(segmentSuffix);
        Mockito.when(quantizationStateReadConfig.getFieldInfo()).thenReturn(fieldInfo);
        Mockito.when(quantizationStateReadConfig.getDirectory()).thenReturn(directory);
        Mockito.when(quantizationStateReadConfig.getScalarQuantizationTypeId()).thenReturn(scalarQuantizationTypeId1);

        try (MockedStatic<KNNQuantizationStateReader> mockedStaticReader = Mockito.mockStatic(KNNQuantizationStateReader.class)) {
            mockedStaticReader.when(() -> KNNQuantizationStateReader.getNumFields(input)).thenReturn(2);
            mockedStaticReader.when(() -> KNNQuantizationStateReader.read(quantizationStateReadConfig)).thenCallRealMethod();
            try (MockedStatic<CodecUtil> mockedStaticCodecUtil = mockStatic(CodecUtil.class)) {
                assertThrows(IllegalArgumentException.class, () -> KNNQuantizationStateReader.read(quantizationStateReadConfig));

                mockedStaticCodecUtil.verify(() -> CodecUtil.retrieveChecksum(input));
                Mockito.verify(input, times(4)).readInt();
                Mockito.verify(input, times(2)).readVLong();
                Mockito.verify(input, times(0)).readBytes(any(byte[].class), anyInt(), anyInt());
                Mockito.verify(input, times(0)).seek(anyLong());

                Mockito.when(input.readInt()).thenReturn(fieldNumber);

                try (MockedStatic<OneBitScalarQuantizationState> mockedStaticOneBit = mockStatic(OneBitScalarQuantizationState.class)) {
                    OneBitScalarQuantizationState oneBitScalarQuantizationState = Mockito.mock(OneBitScalarQuantizationState.class);
                    mockedStaticOneBit.when(() -> OneBitScalarQuantizationState.fromByteArray(any(byte[].class)))
                        .thenReturn(oneBitScalarQuantizationState);
                    QuantizationState quantizationState = KNNQuantizationStateReader.read(quantizationStateReadConfig);
                    assertTrue(quantizationState instanceof OneBitScalarQuantizationState);
                }

                try (MockedStatic<MultiBitScalarQuantizationState> mockedStaticOneBit = mockStatic(MultiBitScalarQuantizationState.class)) {
                    MultiBitScalarQuantizationState multiBitScalarQuantizationState = Mockito.mock(MultiBitScalarQuantizationState.class);
                    mockedStaticOneBit.when(() -> MultiBitScalarQuantizationState.fromByteArray(any(byte[].class)))
                        .thenReturn(multiBitScalarQuantizationState);

                    Mockito.when(quantizationStateReadConfig.getScalarQuantizationTypeId()).thenReturn(scalarQuantizationTypeId2);
                    QuantizationState quantizationState = KNNQuantizationStateReader.read(quantizationStateReadConfig);
                    assertTrue(quantizationState instanceof MultiBitScalarQuantizationState);

                    Mockito.when(quantizationStateReadConfig.getScalarQuantizationTypeId()).thenReturn(scalarQuantizationTypeId4);
                    quantizationState = KNNQuantizationStateReader.read(quantizationStateReadConfig);
                    assertTrue(quantizationState instanceof MultiBitScalarQuantizationState);
                }
                Mockito.when(quantizationStateReadConfig.getScalarQuantizationTypeId()).thenReturn(scalarQuantizationTypeIdIncorrect);
                assertThrows(IllegalArgumentException.class, () -> KNNQuantizationStateReader.read(quantizationStateReadConfig));
            }
        }
    }

    @SneakyThrows
    public void testGetNumFields() {
        IndexInput input = Mockito.mock(IndexInput.class);
        KNNQuantizationStateReader.getNumFields(input);

        Mockito.verify(input, times(2)).readInt();
        Mockito.verify(input, times(1)).readLong();
        Mockito.verify(input, times(2)).seek(anyLong());
        Mockito.verify(input, times(1)).length();
    }
}
