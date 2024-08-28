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
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
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
        String fieldName = "test-field";
        int fieldNumber = 4;
        FieldInfos fieldInfos = Mockito.mock(FieldInfos.class);
        FieldInfo fieldInfo = Mockito.mock(FieldInfo.class);
        Mockito.when(fieldInfo.getFieldNumber()).thenReturn(fieldNumber);
        Mockito.when(fieldInfos.fieldInfo(fieldName)).thenReturn(fieldInfo);

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

        final SegmentReadState segmentReadState = new SegmentReadState(
            directory,
            segmentInfo,
            fieldInfos,
            Mockito.mock(IOContext.class),
            segmentSuffix
        );
        ScalarQuantizationParams scalarQuantizationParams1 = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        ScalarQuantizationParams scalarQuantizationParams2 = new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT);
        ScalarQuantizationParams scalarQuantizationParams4 = new ScalarQuantizationParams(ScalarQuantizationType.FOUR_BIT);
        QuantizationStateReadConfig quantizationStateReadConfig = Mockito.mock(QuantizationStateReadConfig.class);
        Mockito.when(quantizationStateReadConfig.getSegmentReadState()).thenReturn(segmentReadState);
        Mockito.when(quantizationStateReadConfig.getField()).thenReturn(fieldName);

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
                    Mockito.when(quantizationStateReadConfig.getQuantizationParams()).thenReturn(scalarQuantizationParams1);
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

                    Mockito.when(quantizationStateReadConfig.getQuantizationParams()).thenReturn(scalarQuantizationParams2);
                    Mockito.when(quantizationStateReadConfig.getQuantizationParams()).thenReturn(scalarQuantizationParams2);
                    QuantizationState quantizationState = KNNQuantizationStateReader.read(quantizationStateReadConfig);
                    assertTrue(quantizationState instanceof MultiBitScalarQuantizationState);

                    Mockito.when(quantizationStateReadConfig.getQuantizationParams()).thenReturn(scalarQuantizationParams4);
                    Mockito.when(quantizationStateReadConfig.getQuantizationParams()).thenReturn(scalarQuantizationParams4);
                    quantizationState = KNNQuantizationStateReader.read(quantizationStateReadConfig);
                    assertTrue(quantizationState instanceof MultiBitScalarQuantizationState);
                }
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
