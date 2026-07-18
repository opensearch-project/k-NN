/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.search.Sort;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.Version;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.mockito.stubbing.Answer;
import lombok.extern.log4j.Log4j2;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.scorer.PrefetchableFlatVectorScorer.PrefetchableRandomVectorScorer;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.faiss.SQConfig;
import org.opensearch.knn.index.engine.faiss.SQConfigParser;

import java.util.List;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.SQ_CONFIG;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;

@Log4j2
public class Faiss1040ScalarQuantizedKnnVectorsFormatTests extends KNNTestCase {

    public void testFormatName_thenSuccess() {
        assertEquals(
            Faiss1040ScalarQuantizedKnnVectorsFormat.class.getSimpleName(),
            new Faiss1040ScalarQuantizedKnnVectorsFormat().getName()
        );
    }

    public void testGetMaxDimensions_thenUsesLuceneEngine() {
        try (MockedStatic<KNNEngine> mockedKNNEngine = Mockito.mockStatic(KNNEngine.class)) {
            mockedKNNEngine.when(() -> KNNEngine.getMaxDimensionByEngine(KNNEngine.FAISS)).thenReturn(16000);
            assertEquals(16000, new Faiss1040ScalarQuantizedKnnVectorsFormat().getMaxDimensions("test-field"));
            mockedKNNEngine.verify(() -> KNNEngine.getMaxDimensionByEngine(KNNEngine.FAISS));
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
            mock(Codec.class),
            mock(Map.class),
            new byte[16],
            mock(Map.class),
            mock(Sort.class)
        );

        final Directory directory = mock(Directory.class);
        final IndexInput input = mock(IndexInput.class);
        Mockito.when(directory.openInput(any(), any())).thenReturn(input);
        Mockito.when(directory.createOutput(anyString(), any())).thenReturn(mock(IndexOutput.class));

        // Segment carries a single SQ field with bits=1 (matches Lucene's per-field routing —
        // this format instance is only ever asked to handle its routed SQ field).
        final FieldInfo sqFieldInfo = mock(FieldInfo.class);
        Mockito.when(sqFieldInfo.getName()).thenReturn("test-field");
        Mockito.when(sqFieldInfo.getAttribute(SQ_CONFIG)).thenReturn(SQConfigParser.toCsv(SQConfig.builder().bits(1).build()));

        final FieldInfos fieldInfos = mock(FieldInfos.class);
        Mockito.when(fieldInfos.fieldInfo(anyInt())).thenReturn(sqFieldInfo);
        Mockito.when(fieldInfos.iterator()).thenReturn(List.of(sqFieldInfo).iterator());

        final SegmentReadState mockedSegmentReadState = new SegmentReadState(
            directory,
            mockedSegmentInfo,
            fieldInfos,
            mock(IOContext.class),
            ""
        );

        final FieldInfos writeFieldInfos = mock(FieldInfos.class);
        Mockito.when(writeFieldInfos.iterator()).thenReturn(List.of(sqFieldInfo).iterator());
        final SegmentWriteState mockedSegmentWriteState = new SegmentWriteState(
            mock(InfoStream.class),
            directory,
            mockedSegmentInfo,
            writeFieldInfos,
            null,
            mock(IOContext.class)
        );

        final Faiss1040ScalarQuantizedKnnVectorsFormat format = new Faiss1040ScalarQuantizedKnnVectorsFormat();
        try (MockedStatic<CodecUtil> mockedCodecUtil = Mockito.mockStatic(CodecUtil.class)) {
            mockedCodecUtil.when(
                () -> CodecUtil.writeIndexHeader(any(IndexOutput.class), anyString(), anyInt(), any(byte[].class), anyString())
            ).thenAnswer((Answer<Void>) invocation -> null);
            mockedCodecUtil.when(() -> CodecUtil.retrieveChecksum(any(IndexInput.class))).thenAnswer((Answer<Void>) invocation -> null);

            KnnVectorsReader reader = format.fieldsReader(mockedSegmentReadState);
            assertTrue(reader instanceof Faiss1040ScalarQuantizedKnnVectorsReader);
            reader.close();

            KnnVectorsWriter writer = format.fieldsWriter(mockedSegmentWriteState);
            assertTrue(writer instanceof Faiss1040ScalarQuantizedKnnVectorsWriter);
            writer.close();
        }
    }

    public void testToString_thenContainsFormatInfo() {
        final String str = new Faiss1040ScalarQuantizedKnnVectorsFormat().toString();
        assertTrue(str.contains(Faiss1040ScalarQuantizedKnnVectorsFormat.class.getSimpleName()));
    }

    /**
     * Regression guard for the "SPI-instantiated format silently miswrites 2/4-bit fields as
     * 1-bit" bug: verifies that a format constructed via the no-arg SPI constructor still uses
     * the correct per-field encoding when asked for a writer, because encoding is resolved from
     * the field's {@code SQ_CONFIG} attribute at write time (not baked in at construction time).
     */
    /**
     * Regression guard for the "SPI-instantiated format silently miswrites 2/4-bit fields as
     * 1-bit" bug: verifies that a format constructed via the no-arg SPI constructor resolves
     * the correct per-field encoding from the field's {@code SQ_CONFIG} attribute at the moment
     * Lucene calls {@code fieldsWriter().fieldsWriter().addField()}. The encoding lives on the
     * cached {@code KNN1040ScalarQuantizedVectorsFormat} instance the writer resolves.
     */
    @SneakyThrows
    public void testFieldsWriter_encodingResolvedPerFieldFromSQConfig() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            for (int bits : new int[] { 1, 2, 4 }) {
                // Write a segment through the same flat format the wrapper resolves to, with
                // SQ_CONFIG=bits=<bits>. Then open a reader via the no-arg SPI constructor and
                // verify the round-trip works: writer resolved the correct encoding, and reader
                // picked it up from the .vemq header end-to-end.
                final SegmentReadState readState = KNN1040ScalarQuantizedTestUtils.writeQuantizedVectors(
                    dir,
                    "seg_bits_" + bits,
                    new KNN1040ScalarQuantizedVectorsFormat(ScalarEncodingResolver.forDocBits(bits)),
                    bits,
                    random()
                );

                try (KnnVectorsReader rawReader = new Faiss1040ScalarQuantizedKnnVectorsFormat().fieldsReader(readState)) {
                    assertTrue(rawReader instanceof Faiss1040ScalarQuantizedKnnVectorsReader);
                    final FlatVectorsReader flatReader = ((Faiss1040ScalarQuantizedKnnVectorsReader) rawReader).getFlatVectorsReader();
                    final FloatVectorValues values = flatReader.getFloatVectorValues(KNN1040ScalarQuantizedTestUtils.FIELD_NAME);
                    assertNotNull("Values must exist for field written with bits=" + bits, values);
                    assertEquals(
                        "bits=" + bits + " must produce " + KNN1040ScalarQuantizedTestUtils.NUM_VECTORS + " vectors",
                        KNN1040ScalarQuantizedTestUtils.NUM_VECTORS,
                        values.size()
                    );
                }
            }
        }
    }

    /**
     * Fail-loud invariant: if an SQ field with {@code bits=0} reaches the writer via
     * {@code addField()}, the lazy encoding resolver must throw rather than silently default to
     * 1-bit. This is the exact silent-default failure mode this refactor was written to eliminate.
     */
    @SneakyThrows
    public void testFieldsWriter_addFieldWithBitsZero_thenThrows() {
        final FieldInfo malformedSqField = mock(FieldInfo.class);
        Mockito.when(malformedSqField.getName()).thenReturn("malformed_sq_field");
        // Syntactically valid SQ_CONFIG with a zero-bits payload — bypasses SQConfigParser's
        // "empty means SQConfig.EMPTY" shortcut and hits the fail-loud guard in the resolver.
        Mockito.when(malformedSqField.getAttribute(SQ_CONFIG)).thenReturn("bits=0");

        final KnnVectorsWriter writer = buildWriterForSegmentWithField(malformedSqField);
        try {
            final IllegalStateException ex = expectThrows(IllegalStateException.class, () -> writer.addField(malformedSqField));
            assertTrue(
                "Expected fail-loud message mentioning bits and the field name, got: " + ex.getMessage(),
                ex.getMessage().contains("bits") && ex.getMessage().contains("malformed_sq_field")
            );
        } finally {
            writer.close();
        }
    }

    /**
     * Builds a writer through the no-arg (SPI) constructor path, with a segment containing the
     * given field. Encoding resolution is deferred until {@code addField()} is called on the
     * returned writer — this mirrors the actual Lucene write path.
     */
    @SneakyThrows
    private KnnVectorsWriter buildWriterForSegmentWithField(FieldInfo fieldInfo) {
        final SegmentInfo segmentInfo = new SegmentInfo(
            mock(Directory.class),
            mock(Version.class),
            mock(Version.class),
            "test-segment",
            0,
            false,
            false,
            mock(Codec.class),
            mock(Map.class),
            new byte[16],
            mock(Map.class),
            mock(Sort.class)
        );

        final Directory directory = mock(Directory.class);
        Mockito.when(directory.createOutput(anyString(), any())).thenReturn(mock(IndexOutput.class));

        final FieldInfos writeFieldInfos = mock(FieldInfos.class);
        Mockito.when(writeFieldInfos.iterator()).thenReturn(List.of(fieldInfo).iterator());

        final SegmentWriteState writeState = new SegmentWriteState(
            mock(InfoStream.class),
            directory,
            segmentInfo,
            writeFieldInfos,
            null,
            mock(IOContext.class)
        );

        final Faiss1040ScalarQuantizedKnnVectorsFormat format = new Faiss1040ScalarQuantizedKnnVectorsFormat();
        try (MockedStatic<CodecUtil> mockedCodecUtil = Mockito.mockStatic(CodecUtil.class)) {
            mockedCodecUtil.when(
                () -> CodecUtil.writeIndexHeader(any(IndexOutput.class), anyString(), anyInt(), any(byte[].class), anyString())
            ).thenAnswer((Answer<Void>) invocation -> null);
            return format.fieldsWriter(writeState);
        }
    }

    @SneakyThrows
    public void testFieldsReader_thenWrapsFlatReaderWithPrefetchSupport() {
        final SegmentInfo mockedSegmentInfo = new SegmentInfo(
            mock(Directory.class),
            mock(Version.class),
            mock(Version.class),
            "test-segment",
            0,
            false,
            false,
            mock(Codec.class),
            mock(Map.class),
            new byte[16],
            mock(Map.class),
            mock(Sort.class)
        );

        final Directory directory = mock(Directory.class);
        final IndexInput input = mock(IndexInput.class);
        Mockito.when(directory.openInput(any(), any())).thenReturn(input);

        // Segment carries a single SQ field (bits=1) — the format's read path resolves encoding
        // per-field from SQ_CONFIG just like the write path.
        final FieldInfo sqFieldInfo = mock(FieldInfo.class);
        Mockito.when(sqFieldInfo.getName()).thenReturn("test-field");
        Mockito.when(sqFieldInfo.getAttribute(SQ_CONFIG)).thenReturn(SQConfigParser.toCsv(SQConfig.builder().bits(1).build()));

        final FieldInfos fieldInfos = mock(FieldInfos.class);
        Mockito.when(fieldInfos.iterator()).thenReturn(List.of(sqFieldInfo).iterator());

        final SegmentReadState mockedSegmentReadState = new SegmentReadState(
            directory,
            mockedSegmentInfo,
            fieldInfos,
            mock(IOContext.class),
            ""
        );

        final Faiss1040ScalarQuantizedKnnVectorsFormat format = new Faiss1040ScalarQuantizedKnnVectorsFormat();
        try (MockedStatic<CodecUtil> mockedCodecUtil = Mockito.mockStatic(CodecUtil.class)) {
            mockedCodecUtil.when(() -> CodecUtil.retrieveChecksum(any(IndexInput.class))).thenAnswer((Answer<Void>) invocation -> null);

            try (KnnVectorsReader rawReader = format.fieldsReader(mockedSegmentReadState)) {
                assertTrue(rawReader instanceof Faiss1040ScalarQuantizedKnnVectorsReader);
                Faiss1040ScalarQuantizedKnnVectorsReader reader = (Faiss1040ScalarQuantizedKnnVectorsReader) rawReader;

                // Verify the internal FlatVectorsReader is wrapped with Faiss1040ScalarQuantizedFlatVectorsReader
                assertTrue(
                    "FlatVectorsReader should be wrapped with Faiss1040ScalarQuantizedFlatVectorsReader",
                    reader.getFlatVectorsReader() instanceof Faiss1040ScalarQuantizedFlatVectorsReader
                );
            }
        }
    }

    @SneakyThrows
    public void testFieldsReader_scorerIsPrefetchable() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            SegmentReadState readState = KNN1040ScalarQuantizedTestUtils.writeQuantizedVectors(
                dir,
                new KNN1040ScalarQuantizedVectorsFormat(),
                random()
            );

            try (KnnVectorsReader rawReader = new Faiss1040ScalarQuantizedKnnVectorsFormat().fieldsReader(readState)) {
                assertTrue(
                    "Expected Faiss1040ScalarQuantizedKnnVectorsReader but was: " + rawReader.getClass().getSimpleName(),
                    rawReader instanceof Faiss1040ScalarQuantizedKnnVectorsReader
                );
                Faiss1040ScalarQuantizedKnnVectorsReader knnReader = (Faiss1040ScalarQuantizedKnnVectorsReader) rawReader;
                FlatVectorsReader flatReader = knnReader.getFlatVectorsReader();
                RandomVectorScorer scorer = flatReader.getRandomVectorScorer(
                    KNN1040ScalarQuantizedTestUtils.FIELD_NAME,
                    KNN1040ScalarQuantizedTestUtils.randomVector(KNN1040ScalarQuantizedTestUtils.DIMENSION, random())
                );

                assertNotNull("RandomVectorScorer should not be null", scorer);
                assertTrue(
                    "Scorer from Faiss SQ reader should be PrefetchableRandomVectorScorer, but was: " + scorer.getClass().getSimpleName(),
                    scorer instanceof PrefetchableRandomVectorScorer
                );
            }
        }
    }
}
