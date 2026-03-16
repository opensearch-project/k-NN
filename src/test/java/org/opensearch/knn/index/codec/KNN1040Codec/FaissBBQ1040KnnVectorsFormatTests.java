/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.search.Sort;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.Version;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.mockito.stubbing.Answer;
import lombok.extern.log4j.Log4j2;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.engine.KNNEngine;

import java.util.Iterator;
import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;

@Log4j2
public class FaissBBQ1040KnnVectorsFormatTests extends KNNTestCase {

    public void testFormatName_thenSuccess() {
        assertEquals("FaissBBQ1040KnnVectorsFormat", new FaissBBQ1040KnnVectorsFormat().getName());
    }

    public void testDefaultConstructor_thenUsesDefaultThreshold() {
        assertTrue(
            new FaissBBQ1040KnnVectorsFormat().toString()
                .contains("approximateThreshold=" + KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE)
        );
    }

    public void testParameterizedConstructor_thenUsesProvidedThreshold() {
        assertTrue(new FaissBBQ1040KnnVectorsFormat(42).toString().contains("approximateThreshold=42"));
    }

    public void testGetMaxDimensions_thenUsesLuceneEngine() {
        try (MockedStatic<KNNEngine> mockedKNNEngine = Mockito.mockStatic(KNNEngine.class)) {
            mockedKNNEngine.when(() -> KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE)).thenReturn(16000);
            assertEquals(16000, new FaissBBQ1040KnnVectorsFormat().getMaxDimensions("test-field"));
            mockedKNNEngine.verify(() -> KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE));
        }
    }

    @SneakyThrows
    public void testReaderAndWriter_whenValidInput_thenSuccess() {
        final SegmentInfo mockedSegmentInfo = new SegmentInfo(
            mock(Directory.class),
            mock(Version.class),
            mock(Version.class),
            "test-segment",
            0,
            false,
            false,
            mock(org.apache.lucene.codecs.Codec.class),
            mock(Map.class),
            new byte[16],
            mock(Map.class),
            mock(Sort.class)
        );

        final Directory directory = mock(Directory.class);
        final IndexInput input = mock(IndexInput.class);
        Mockito.when(directory.openInput(any(), any())).thenReturn(input);
        Mockito.when(directory.createOutput(anyString(), any())).thenReturn(mock(IndexOutput.class));

        final FieldInfos fieldInfos = mock(FieldInfos.class);
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        Mockito.when(fieldInfo.getName()).thenReturn("test-field");
        Mockito.when(fieldInfos.fieldInfo(anyInt())).thenReturn(fieldInfo);
        Mockito.when(fieldInfos.iterator()).thenReturn(new Iterator<FieldInfo>() {
            @Override
            public boolean hasNext() {
                return false;
            }

            @Override
            public FieldInfo next() {
                return null;
            }
        });

        final SegmentReadState mockedSegmentReadState = new SegmentReadState(
            directory,
            mockedSegmentInfo,
            fieldInfos,
            mock(IOContext.class),
            "test-segment-suffix"
        );

        final SegmentWriteState mockedSegmentWriteState = new SegmentWriteState(
            mock(InfoStream.class),
            directory,
            mockedSegmentInfo,
            mock(FieldInfos.class),
            null,
            mock(IOContext.class)
        );

        final FaissBBQ1040KnnVectorsFormat format = new FaissBBQ1040KnnVectorsFormat(0);
        try (MockedStatic<CodecUtil> mockedCodecUtil = Mockito.mockStatic(CodecUtil.class)) {
            mockedCodecUtil.when(
                () -> CodecUtil.writeIndexHeader(any(IndexOutput.class), anyString(), anyInt(), any(byte[].class), anyString())
            ).thenAnswer((Answer<Void>) invocation -> null);
            mockedCodecUtil.when(() -> CodecUtil.retrieveChecksum(any(IndexInput.class))).thenAnswer((Answer<Void>) invocation -> null);

            KnnVectorsReader reader = format.fieldsReader(mockedSegmentReadState);
            assertTrue(reader instanceof FaissBBQ1040KnnVectorsReader);
            reader.close();

            KnnVectorsWriter writer = format.fieldsWriter(mockedSegmentWriteState);
            assertTrue(writer instanceof FaissBBQ1040KnnVectorsWriter);
            writer.close();
        }
    }

    public void testApproximateThreshold_whenMultipleInstances_thenIndependent() {
        assertTrue(new FaissBBQ1040KnnVectorsFormat(100).toString().contains("approximateThreshold=100"));
        assertTrue(new FaissBBQ1040KnnVectorsFormat(200).toString().contains("approximateThreshold=200"));
    }

    public void testToString_thenContainsFormatInfo() {
        final String str = new FaissBBQ1040KnnVectorsFormat(50).toString();
        assertTrue(str.contains("FaissBBQ1040KnnVectorsFormat"));
        assertTrue(str.contains("approximateThreshold=50"));
    }
}
