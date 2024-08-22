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
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.search.Sort;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Version;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;

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

        final SegmentReadState segmentReadState = new SegmentReadState(
            directory,
            segmentInfo,
            Mockito.mock(FieldInfos.class),
            Mockito.mock(IOContext.class),
            segmentSuffix
        );

        KNNQuantizationStateReader quantizationStateReader = Mockito.mock(KNNQuantizationStateReader.class);
        Mockito.when(quantizationStateReader.getNumFields(input)).thenReturn(2);
        Mockito.when(quantizationStateReader.read(segmentReadState)).thenCallRealMethod();

        try (MockedStatic<CodecUtil> mockedStaticCodecUtil = mockStatic(CodecUtil.class)) {
            quantizationStateReader.read(segmentReadState);

            mockedStaticCodecUtil.verify(() -> CodecUtil.retrieveChecksum(input));
            Mockito.verify(input, times(2)).readInt();
            Mockito.verify(input, times(2)).readString();
            Mockito.verify(input, times(2)).readVLong();
            Mockito.verify(input, times(2)).readBytes(any(byte[].class), anyInt(), anyInt());
            Mockito.verify(input, times(2)).seek(anyLong());
        }
    }

    @SneakyThrows
    public void testGetNumFields() {
        IndexInput input = Mockito.mock(IndexInput.class);
        KNNQuantizationStateReader quantizationStateReader = new KNNQuantizationStateReader();
        quantizationStateReader.getNumFields(input);

        Mockito.verify(input, times(2)).readInt();
        Mockito.verify(input, times(1)).readLong();
        Mockito.verify(input, times(2)).seek(anyLong());
        Mockito.verify(input, times(1)).length();
    }
}
