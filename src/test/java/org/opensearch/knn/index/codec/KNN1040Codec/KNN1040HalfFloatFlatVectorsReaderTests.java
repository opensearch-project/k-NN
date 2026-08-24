/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfHalfFloatsSerializer;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Tests {@link KNN1040HalfFloatFlatVectorsReader} in isolation, without going through
 * {@link KNN1040HalfFloatFlatVectorsWriter}. Fixtures are written directly with low-level Lucene
 * primitives (mirroring the writer's on-disk layout, not calling the writer class itself), so a bug
 * introduced only in the writer can't mask a bug in the reader, or vice versa.
 */
public class KNN1040HalfFloatFlatVectorsReaderTests extends KNNTestCase {

    private static final String FIELD_NAME = "fp16_vector";
    private static final int DIMENSION = 4;
    private static final int VECTOR_DATA_ALIGNMENT = 64;

    @SneakyThrows
    public void testGetFloatVectorValues_decodesCorrectly() {
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] vectors = { { 1.5f, -2.5f, 3.25f, 0.0f }, { -1.0f, 2.0f, -3.0f, 4.0f }, { 0.5f, 0.5f, 0.5f, 0.5f } };
            SegmentReadState readState = writeRawSegment(dir, vectors, VectorSimilarityFunction.EUCLIDEAN);

            try (FlatVectorsReader reader = newReader(readState)) {
                FloatVectorValues values = reader.getFloatVectorValues(FIELD_NAME);
                assertNotNull(values);
                assertEquals(DIMENSION, values.dimension());
                assertEquals(vectors.length, values.size());

                for (int i = 0; i < vectors.length; i++) {
                    float[] actual = values.vectorValue(i);
                    for (int d = 0; d < DIMENSION; d++) {
                        float expected = Float.float16ToFloat(Float.floatToFloat16(vectors[i][d]));
                        assertEquals("vector " + i + " dim " + d, expected, actual[d], 0.0f);
                    }
                }
            }
        }
    }

    @SneakyThrows
    public void testGetByteVectorValues_throws() {
        try (Directory dir = new ByteBuffersDirectory()) {
            SegmentReadState readState = writeRawSegment(dir, new float[][] { { 1f, 2f, 3f, 4f } }, VectorSimilarityFunction.EUCLIDEAN);
            try (FlatVectorsReader reader = newReader(readState)) {
                expectThrows(UnsupportedOperationException.class, () -> reader.getByteVectorValues(FIELD_NAME));
                expectThrows(UnsupportedOperationException.class, () -> reader.getRandomVectorScorer(FIELD_NAME, new byte[] { 1 }));
            }
        }
    }

    @SneakyThrows
    public void testGetFloatVectorValues_unknownField_throws() {
        try (Directory dir = new ByteBuffersDirectory()) {
            SegmentReadState readState = writeRawSegment(dir, new float[][] { { 1f, 2f, 3f, 4f } }, VectorSimilarityFunction.EUCLIDEAN);
            try (FlatVectorsReader reader = newReader(readState)) {
                IllegalArgumentException e = expectThrows(IllegalArgumentException.class, () -> reader.getFloatVectorValues("nope"));
                assertTrue(e.getMessage().contains("nope"));
            }
        }
    }

    @SneakyThrows
    public void testCheckIntegrity_validSegment_succeeds() {
        try (Directory dir = new ByteBuffersDirectory()) {
            SegmentReadState readState = writeRawSegment(dir, new float[][] { { 1f, 2f, 3f, 4f } }, VectorSimilarityFunction.EUCLIDEAN);
            try (FlatVectorsReader reader = newReader(readState)) {
                reader.checkIntegrity();
            }
        }
    }

    @SneakyThrows
    public void testGetOffHeapByteSize_returnsVectorDataLength() {
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] vectors = { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f } };
            SegmentReadState readState = writeRawSegment(dir, vectors, VectorSimilarityFunction.EUCLIDEAN);
            try (FlatVectorsReader reader = newReader(readState)) {
                FieldInfo fieldInfo = readState.fieldInfos.fieldInfo(FIELD_NAME);
                Map<String, Long> offHeap = reader.getOffHeapByteSize(fieldInfo);
                long expectedBytes = (long) vectors.length * DIMENSION * Short.BYTES;
                assertEquals(Long.valueOf(expectedBytes), offHeap.get(KNN1040HalfFloatFlatVectorsFormat.VECTOR_DATA_EXTENSION));
            }
        }
    }

    @SneakyThrows
    public void testRamBytesUsed_positive() {
        try (Directory dir = new ByteBuffersDirectory()) {
            SegmentReadState readState = writeRawSegment(dir, new float[][] { { 1f, 2f, 3f, 4f } }, VectorSimilarityFunction.EUCLIDEAN);
            try (FlatVectorsReader reader = newReader(readState)) {
                assertTrue(reader.ramBytesUsed() > 0);
            }
        }
    }

    @SneakyThrows
    public void testFieldEntry_dimensionMismatch_throws() {
        try (Directory dir = new ByteBuffersDirectory()) {
            // Meta records DIMENSION+1 vectors' worth of dimension, but FieldInfo (read separately by
            // the reader) still says DIMENSION - triggers the "Inconsistent vector dimension" check.
            SegmentReadState readState = writeRawSegment(
                dir,
                new float[][] { { 1f, 2f, 3f, 4f } },
                VectorSimilarityFunction.EUCLIDEAN,
                Overrides.builder().metaDimension(DIMENSION + 1).build()
            );
            IllegalStateException e = expectThrows(IllegalStateException.class, () -> newReader(readState));
            assertTrue(e.getMessage().contains("Inconsistent vector dimension"));
        }
    }

    @SneakyThrows
    public void testFieldEntry_similarityMismatch_throws() {
        try (Directory dir = new ByteBuffersDirectory()) {
            // Meta records COSINE's ordinal, but FieldInfo says EUCLIDEAN - triggers the "Inconsistent
            // vector similarity function" check.
            SegmentReadState readState = writeRawSegment(
                dir,
                new float[][] { { 1f, 2f, 3f, 4f } },
                VectorSimilarityFunction.EUCLIDEAN,
                Overrides.builder().metaSimilarityOrdinal(VectorSimilarityFunction.COSINE.ordinal()).build()
            );
            IllegalStateException e = expectThrows(IllegalStateException.class, () -> newReader(readState));
            assertTrue(e.getMessage().contains("Inconsistent vector similarity function"));
        }
    }

    @SneakyThrows
    public void testFieldEntry_encodingMismatch_throws() {
        try (Directory dir = new ByteBuffersDirectory()) {
            SegmentReadState readState = writeRawSegment(
                dir,
                new float[][] { { 1f, 2f, 3f, 4f } },
                VectorSimilarityFunction.EUCLIDEAN,
                Overrides.builder().metaEncodingOrdinal(VectorEncoding.BYTE.ordinal()).build()
            );
            IllegalStateException e = expectThrows(IllegalStateException.class, () -> newReader(readState));
            assertTrue(e.getMessage().contains("Unexpected vector encoding"));
        }
    }

    @SneakyThrows
    public void testFieldEntry_lengthMismatch_throws() {
        try (Directory dir = new ByteBuffersDirectory()) {
            SegmentReadState readState = writeRawSegment(
                dir,
                new float[][] { { 1f, 2f, 3f, 4f } },
                VectorSimilarityFunction.EUCLIDEAN,
                Overrides.builder().metaVectorDataLength(1).build()
            );
            IllegalStateException e = expectThrows(IllegalStateException.class, () -> newReader(readState));
            assertTrue(e.getMessage().contains("Vector data length"));
        }
    }

    private FlatVectorsReader newReader(SegmentReadState readState) throws Exception {
        FlatVectorsScorer scorer = Mockito.mock(FlatVectorsScorer.class);
        return new KNN1040HalfFloatFlatVectorsReader(readState, scorer);
    }

    /** Optional per-field overrides used to write a deliberately corrupted fixture. */
    private static final class Overrides {
        private final Integer metaDimension;
        private final Integer metaSimilarityOrdinal;
        private final Integer metaEncodingOrdinal;
        private final Long metaVectorDataLength;

        private Overrides(Integer metaDimension, Integer metaSimilarityOrdinal, Integer metaEncodingOrdinal, Long metaVectorDataLength) {
            this.metaDimension = metaDimension;
            this.metaSimilarityOrdinal = metaSimilarityOrdinal;
            this.metaEncodingOrdinal = metaEncodingOrdinal;
            this.metaVectorDataLength = metaVectorDataLength;
        }

        static Builder builder() {
            return new Builder();
        }

        static final class Builder {
            private Integer metaDimension;
            private Integer metaSimilarityOrdinal;
            private Integer metaEncodingOrdinal;
            private Long metaVectorDataLength;

            Builder metaDimension(int v) {
                this.metaDimension = v;
                return this;
            }

            Builder metaSimilarityOrdinal(int v) {
                this.metaSimilarityOrdinal = v;
                return this;
            }

            Builder metaEncodingOrdinal(int v) {
                this.metaEncodingOrdinal = v;
                return this;
            }

            Builder metaVectorDataLength(long v) {
                this.metaVectorDataLength = v;
                return this;
            }

            Overrides build() {
                return new Overrides(metaDimension, metaSimilarityOrdinal, metaEncodingOrdinal, metaVectorDataLength);
            }
        }
    }

    private SegmentReadState writeRawSegment(Directory dir, float[][] vectors, VectorSimilarityFunction similarity) throws Exception {
        return writeRawSegment(dir, vectors, similarity, Overrides.builder().build());
    }

    /**
     * Hand-writes a segment's {@code .vemf}/{@code .vec} pair using the same low-level primitives and
     * on-disk layout as {@link KNN1040HalfFloatFlatVectorsWriter}, without depending on that class.
     * {@code overrides} lets individual tests corrupt one recorded value to exercise
     * {@code FieldEntry}'s validation checks.
     */
    private SegmentReadState writeRawSegment(
        Directory dir,
        float[][] vectors,
        VectorSimilarityFunction similarity,
        Overrides overrides
    ) throws Exception {
        FieldInfo fieldInfo = createFieldInfo(similarity);
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });
        SegmentInfo segmentInfo = new SegmentInfo(
            dir,
            Version.LATEST,
            Version.LATEST,
            "_0",
            vectors.length,
            false,
            false,
            null,
            Collections.emptyMap(),
            StringHelper.randomId(),
            new HashMap<>(),
            null
        );

        String metaFileName = IndexFileNames.segmentFileName("_0", "", KNN1040HalfFloatFlatVectorsFormat.META_EXTENSION);
        String vectorDataFileName = IndexFileNames.segmentFileName("_0", "", KNN1040HalfFloatFlatVectorsFormat.VECTOR_DATA_EXTENSION);

        try (
            IndexOutput meta = dir.createOutput(metaFileName, IOContext.DEFAULT);
            IndexOutput vectorData = dir.createOutput(vectorDataFileName, IOContext.DEFAULT)
        ) {
            CodecUtil.writeIndexHeader(
                meta,
                KNN1040HalfFloatFlatVectorsFormat.META_CODEC_NAME,
                KNN1040HalfFloatFlatVectorsFormat.VERSION_CURRENT,
                segmentInfo.getId(),
                ""
            );
            CodecUtil.writeIndexHeader(
                vectorData,
                KNN1040HalfFloatFlatVectorsFormat.VECTOR_DATA_CODEC_NAME,
                KNN1040HalfFloatFlatVectorsFormat.VERSION_CURRENT,
                segmentInfo.getId(),
                ""
            );

            long vectorDataOffset = vectorData.alignFilePointer(VECTOR_DATA_ALIGNMENT);
            byte[] outputBuffer = new byte[DIMENSION * Short.BYTES];
            DocsWithFieldSet docsWithField = new DocsWithFieldSet();
            for (int doc = 0; doc < vectors.length; doc++) {
                KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE.floatToByteArray(vectors[doc], outputBuffer, DIMENSION);
                vectorData.writeBytes(outputBuffer, 0, outputBuffer.length);
                docsWithField.add(doc);
            }
            long vectorDataLength = vectorData.getFilePointer() - vectorDataOffset;

            meta.writeInt(fieldInfo.number);
            meta.writeInt(overrides.metaEncodingOrdinal != null ? overrides.metaEncodingOrdinal : VectorEncoding.FLOAT32.ordinal());
            meta.writeInt(overrides.metaSimilarityOrdinal != null ? overrides.metaSimilarityOrdinal : similarity.ordinal());
            meta.writeVLong(vectorDataOffset);
            meta.writeVLong(overrides.metaVectorDataLength != null ? overrides.metaVectorDataLength : vectorDataLength);
            meta.writeVInt(overrides.metaDimension != null ? overrides.metaDimension : DIMENSION);

            int count = docsWithField.cardinality();
            meta.writeInt(count);
            OrdToDocDISIReaderConfiguration.writeStoredMeta(
                KNN1040HalfFloatFlatVectorsFormat.DIRECT_MONOTONIC_BLOCK_SHIFT,
                meta,
                vectorData,
                count,
                vectors.length,
                docsWithField
            );

            meta.writeInt(-1);
            CodecUtil.writeFooter(meta);
            CodecUtil.writeFooter(vectorData);
        }

        return new SegmentReadState(dir, segmentInfo, fieldInfos, IOContext.DEFAULT);
    }

    private FieldInfo createFieldInfo(VectorSimilarityFunction similarity) {
        return new FieldInfo(
            FIELD_NAME,
            0,
            false,
            false,
            false,
            IndexOptions.NONE,
            DocValuesType.NONE,
            DocValuesSkipIndexType.NONE,
            -1,
            Map.of(),
            0,
            0,
            0,
            DIMENSION,
            VectorEncoding.FLOAT32,
            similarity,
            false,
            false
        );
    }
}
