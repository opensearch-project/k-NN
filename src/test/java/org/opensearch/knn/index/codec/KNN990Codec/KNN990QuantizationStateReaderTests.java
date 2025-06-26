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
import org.apache.lucene.index.IndexFileNames;
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
import org.opensearch.knn.common.KNNConstants;
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

public class KNN990QuantizationStateReaderTests extends KNNTestCase {

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
        ScalarQuantizationParams scalarQuantizationParams1 = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build();
        ScalarQuantizationParams scalarQuantizationParams2 = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.TWO_BIT).build();
        ScalarQuantizationParams scalarQuantizationParams4 = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.FOUR_BIT).build();
        QuantizationStateReadConfig quantizationStateReadConfig = Mockito.mock(QuantizationStateReadConfig.class);
        Mockito.when(quantizationStateReadConfig.getSegmentReadState()).thenReturn(segmentReadState);
        Mockito.when(quantizationStateReadConfig.getField()).thenReturn(fieldName);

        try (MockedStatic<KNN990QuantizationStateReader> mockedStaticReader = Mockito.mockStatic(KNN990QuantizationStateReader.class)) {
            mockedStaticReader.when(() -> KNN990QuantizationStateReader.getNumFields(input)).thenReturn(2);
            mockedStaticReader.when(() -> KNN990QuantizationStateReader.read(quantizationStateReadConfig)).thenCallRealMethod();
            mockedStaticReader.when(() -> KNN990QuantizationStateReader.readStateBytes(any(IndexInput.class), anyLong(), anyInt()))
                .thenReturn(new byte[8]);
            try (MockedStatic<CodecUtil> mockedStaticCodecUtil = mockStatic(CodecUtil.class)) {

                Mockito.when(input.readInt()).thenReturn(fieldNumber);

                try (MockedStatic<OneBitScalarQuantizationState> mockedStaticOneBit = mockStatic(OneBitScalarQuantizationState.class)) {
                    Mockito.when(quantizationStateReadConfig.getQuantizationParams()).thenReturn(scalarQuantizationParams1);
                    OneBitScalarQuantizationState oneBitScalarQuantizationState = Mockito.mock(OneBitScalarQuantizationState.class);
                    mockedStaticOneBit.when(() -> OneBitScalarQuantizationState.fromByteArray(any(byte[].class)))
                        .thenReturn(oneBitScalarQuantizationState);
                    QuantizationState quantizationState = KNN990QuantizationStateReader.read(quantizationStateReadConfig);
                    assertEquals(oneBitScalarQuantizationState, quantizationState);
                    mockedStaticCodecUtil.verify(() -> CodecUtil.retrieveChecksum(input));
                }

                try (MockedStatic<MultiBitScalarQuantizationState> mockedStaticOneBit = mockStatic(MultiBitScalarQuantizationState.class)) {
                    MultiBitScalarQuantizationState multiBitScalarQuantizationState = Mockito.mock(MultiBitScalarQuantizationState.class);
                    mockedStaticOneBit.when(() -> MultiBitScalarQuantizationState.fromByteArray(any(byte[].class)))
                        .thenReturn(multiBitScalarQuantizationState);

                    Mockito.when(quantizationStateReadConfig.getQuantizationParams()).thenReturn(scalarQuantizationParams2);
                    Mockito.when(quantizationStateReadConfig.getQuantizationParams()).thenReturn(scalarQuantizationParams2);
                    QuantizationState quantizationState = KNN990QuantizationStateReader.read(quantizationStateReadConfig);
                    assertEquals(multiBitScalarQuantizationState, quantizationState);

                    Mockito.when(quantizationStateReadConfig.getQuantizationParams()).thenReturn(scalarQuantizationParams4);
                    Mockito.when(quantizationStateReadConfig.getQuantizationParams()).thenReturn(scalarQuantizationParams4);
                    quantizationState = KNN990QuantizationStateReader.read(quantizationStateReadConfig);
                    assertEquals(multiBitScalarQuantizationState, quantizationState);
                }
            }
        }
    }

    @SneakyThrows
    public void testGetNumFields() {
        IndexInput input = Mockito.mock(IndexInput.class);
        KNN990QuantizationStateReader.getNumFields(input);

        Mockito.verify(input, times(1)).readInt();
        Mockito.verify(input, times(1)).readLong();
        Mockito.verify(input, times(2)).seek(anyLong());
        Mockito.verify(input, times(1)).length();
    }

    @SneakyThrows
    public void testReadStateBytes() {
        IndexInput input = Mockito.mock(IndexInput.class);
        long position = 1;
        int length = 2;
        byte[] stateBytes = new byte[length];
        KNN990QuantizationStateReader.readStateBytes(input, position, length);

        Mockito.verify(input, times(1)).seek(position);
        Mockito.verify(input, times(1)).readBytes(stateBytes, 0, length);

    }

    @SneakyThrows
    public void testGetQuantizationStateFileName() {
        String segmentName = "test-segment";
        String segmentSuffix = "test-suffix";
        String expectedName = IndexFileNames.segmentFileName(segmentName, segmentSuffix, KNNConstants.QUANTIZATION_STATE_FILE_SUFFIX);

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

        final SegmentReadState segmentReadState = new SegmentReadState(
            Mockito.mock(Directory.class),
            segmentInfo,
            Mockito.mock(FieldInfos.class),
            Mockito.mock(IOContext.class),
            segmentSuffix
        );

        assertEquals(expectedName, KNN990QuantizationStateReader.getQuantizationStateFileName(segmentReadState));

    }
}
