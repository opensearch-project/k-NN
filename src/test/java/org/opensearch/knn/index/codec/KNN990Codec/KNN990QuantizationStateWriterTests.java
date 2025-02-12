/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.search.Sort;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.Version;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.mockito.stubbing.Answer;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.times;

public class KNN990QuantizationStateWriterTests extends KNNTestCase {

    @SneakyThrows
    public void testWriteHeader() {
        final String segmentName = "test-segment-name";

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
        IndexOutput output = Mockito.mock(IndexOutput.class);
        Mockito.when(directory.createOutput(any(), any())).thenReturn(output);

        final SegmentWriteState segmentWriteState = new SegmentWriteState(
            Mockito.mock(InfoStream.class),
            directory,
            segmentInfo,
            Mockito.mock(FieldInfos.class),
            null,
            Mockito.mock(IOContext.class)
        );
        KNN990QuantizationStateWriter quantizationStateWriter = new KNN990QuantizationStateWriter(segmentWriteState);
        try (MockedStatic<CodecUtil> mockedStaticCodecUtil = Mockito.mockStatic(CodecUtil.class)) {
            mockedStaticCodecUtil.when(
                () -> CodecUtil.writeIndexHeader(any(IndexOutput.class), anyString(), anyInt(), any(byte[].class), anyString())
            ).thenAnswer((Answer<Void>) invocation -> null);
            quantizationStateWriter.writeHeader(segmentWriteState);
            mockedStaticCodecUtil.verify(
                () -> CodecUtil.writeIndexHeader(
                    output,
                    KNN990QuantizationStateWriter.NATIVE_ENGINES_990_KNN_VECTORS_FORMAT_QS_DATA,
                    0,
                    segmentWriteState.segmentInfo.getId(),
                    segmentWriteState.segmentSuffix
                )
            );
        }
    }

    @SneakyThrows
    public void testWriteState() {
        final String segmentName = "test-segment-name";

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
        IndexOutput output = Mockito.mock(IndexOutput.class);
        Mockito.when(directory.createOutput(any(), any())).thenReturn(output);

        final SegmentWriteState segmentWriteState = new SegmentWriteState(
            Mockito.mock(InfoStream.class),
            directory,
            segmentInfo,
            Mockito.mock(FieldInfos.class),
            null,
            Mockito.mock(IOContext.class)
        );
        KNN990QuantizationStateWriter quantizationStateWriter = new KNN990QuantizationStateWriter(segmentWriteState);

        int fieldNumber = 0;
        QuantizationState quantizationState = OneBitScalarQuantizationState.builder()
            .quantizationParams(new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT))
            .meanThresholds(new float[] { 1.2f, 2.3f, 3.4f, 4.5f })
            .build();
        quantizationStateWriter.writeState(fieldNumber, quantizationState);
        byte[] stateBytes = quantizationState.toByteArray();
        Mockito.verify(output, times(1)).writeBytes(stateBytes, stateBytes.length);
    }

    @SneakyThrows
    public void testWriteFooter() {
        final String segmentName = "test-segment-name";

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
        IndexOutput output = Mockito.mock(IndexOutput.class);
        Mockito.when(directory.createOutput(any(), any())).thenReturn(output);

        final SegmentWriteState segmentWriteState = new SegmentWriteState(
            Mockito.mock(InfoStream.class),
            directory,
            segmentInfo,
            Mockito.mock(FieldInfos.class),
            null,
            Mockito.mock(IOContext.class)
        );
        KNN990QuantizationStateWriter quantizationStateWriter = new KNN990QuantizationStateWriter(segmentWriteState);

        int fieldNumber1 = 1;
        int fieldNumber2 = 2;
        QuantizationState quantizationState1 = OneBitScalarQuantizationState.builder()
            .quantizationParams(new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT))
            .meanThresholds(new float[] { 1.2f, 2.3f, 3.4f, 4.5f })
            .build();
        QuantizationState quantizationState2 = OneBitScalarQuantizationState.builder()
            .quantizationParams(new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT))
            .meanThresholds(new float[] { 2.3f, 3.4f, 4.5f, 5.6f })
            .build();
        quantizationStateWriter.writeState(fieldNumber1, quantizationState1);
        quantizationStateWriter.writeState(fieldNumber2, quantizationState2);

        try (MockedStatic<CodecUtil> mockedStaticCodecUtil = mockStatic(CodecUtil.class)) {
            quantizationStateWriter.writeFooter();

            Mockito.verify(output, times(6)).writeInt(anyInt());
            Mockito.verify(output, times(2)).writeVLong(anyLong());
            Mockito.verify(output, times(1)).writeLong(anyLong());
            mockedStaticCodecUtil.verify(() -> CodecUtil.writeFooter(output));
        }
    }
}
