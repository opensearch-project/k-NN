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
import org.apache.lucene.index.CorruptIndexException;
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
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.scorer.NativeEngines990KnnVectorsScorer;
import org.opensearch.knn.index.codec.scorer.PrefetchableFlatVectorScorer;
import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfHalfFloatsSerializer;
import org.opensearch.knn.memoryoptsearch.faiss.FlatVectorsScorerProvider;
import org.opensearch.knn.memoryoptsearch.faiss.MMapFloatVectorValues;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

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

    @SneakyThrows
    public void testReadFields_unknownFieldNumber_throws() {
        try (Directory dir = new ByteBuffersDirectory()) {
            SegmentReadState readState = writeRawSegment(
                dir,
                new float[][] { { 1f, 2f, 3f, 4f } },
                VectorSimilarityFunction.EUCLIDEAN,
                Overrides.builder().metaFieldNumber(99).build()
            );
            CorruptIndexException e = expectThrows(CorruptIndexException.class, () -> newReader(readState));
            assertTrue(e.getMessage().contains("Invalid field number"));
        }
    }

    @SneakyThrows
    public void testReadSimilarityFunction_outOfRangeOrdinal_throws() {
        try (Directory dir = new ByteBuffersDirectory()) {
            SegmentReadState readState = writeRawSegment(
                dir,
                new float[][] { { 1f, 2f, 3f, 4f } },
                VectorSimilarityFunction.EUCLIDEAN,
                Overrides.builder().metaSimilarityOrdinal(99).build()
            );
            IllegalArgumentException e = expectThrows(IllegalArgumentException.class, () -> newReader(readState));
            assertTrue(e.getMessage().contains("invalid distance function"));
        }
    }

    @SneakyThrows
    public void testReadVectorEncoding_outOfRangeOrdinal_throws() {
        try (Directory dir = new ByteBuffersDirectory()) {
            SegmentReadState readState = writeRawSegment(
                dir,
                new float[][] { { 1f, 2f, 3f, 4f } },
                VectorSimilarityFunction.EUCLIDEAN,
                Overrides.builder().metaEncodingOrdinal(99).build()
            );
            CorruptIndexException e = expectThrows(CorruptIndexException.class, () -> newReader(readState));
            assertTrue(e.getMessage().contains("Invalid vector encoding id"));
        }
    }

    @SneakyThrows
    public void testSparseDocs_ordToDocMappingUsesDirectMonotonicReader() {
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] vectors = { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f }, { 9f, 10f, 11f, 12f } };
            int[] docIds = { 0, 3, 7 };
            SegmentReadState readState = writeRawSegment(
                dir,
                vectors,
                VectorSimilarityFunction.EUCLIDEAN,
                Overrides.builder().docIds(docIds).maxDoc(10).build()
            );

            try (FlatVectorsReader reader = newReader(readState)) {
                FloatVectorValues values = reader.getFloatVectorValues(FIELD_NAME);
                for (int ord = 0; ord < docIds.length; ord++) {
                    assertEquals(docIds[ord], values.ordToDoc(ord));
                }
                FloatVectorValues.DocIndexIterator iterator = values.iterator();
                int ord = 0;
                for (int doc = iterator.nextDoc(); doc != org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS; doc = iterator
                    .nextDoc()) {
                    assertEquals(docIds[ord], doc);
                    assertEquals(ord, iterator.index());
                    ord++;
                }
                assertEquals(docIds.length, ord);
            }
        }
    }

    @SneakyThrows
    public void testGetMergeInstance_returnsUsableReaderThenFinishMergeRestoresIt() {
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] vectors = { { 1f, 2f, 3f, 4f } };
            SegmentReadState readState = writeRawSegment(dir, vectors, VectorSimilarityFunction.EUCLIDEAN);

            try (FlatVectorsReader reader = newReader(readState)) {
                FlatVectorsReader mergeInstance = reader.getMergeInstance();
                assertSame(reader, mergeInstance);
                assertEquals(1, mergeInstance.getFloatVectorValues(FIELD_NAME).size());
                mergeInstance.finishMerge();
                assertEquals(1, reader.getFloatVectorValues(FIELD_NAME).size());
            }
        }
    }

    @SneakyThrows
    public void testGetRandomVectorScorer_nonMmapDirectory_matchesExpectedSimilarity() {
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] vectors = { { 1.5f, -2.5f, 3.25f, 0.0f }, { -1.0f, 2.0f, -3.0f, 4.0f }, { 0.5f, 0.5f, 0.5f, 0.5f } };
            SegmentReadState readState = writeRawSegment(dir, vectors, VectorSimilarityFunction.EUCLIDEAN);

            try (FlatVectorsReader reader = newReader(readState)) {
                float[] query = { 1f, 1f, 1f, 1f };
                RandomVectorScorer scorer = reader.getRandomVectorScorer(FIELD_NAME, query);
                assertNotNull(scorer);
                assertEquals(vectors.length, scorer.maxOrd());

                for (int i = 0; i < vectors.length; i++) {
                    float[] decoded = new float[DIMENSION];
                    for (int d = 0; d < DIMENSION; d++) {
                        decoded[d] = Float.float16ToFloat(Float.floatToFloat16(vectors[i][d]));
                    }
                    float expected = VectorSimilarityFunction.EUCLIDEAN.compare(query, decoded);
                    assertEquals("Vector " + i, expected, scorer.score(i), 1e-3f);
                }
            }
        }
    }

    @SneakyThrows
    public void testGetRandomVectorScorer_emptySegment_returnsNull() {
        try (Directory dir = new ByteBuffersDirectory()) {
            SegmentReadState readState = writeRawSegment(
                dir,
                new float[0][0],
                VectorSimilarityFunction.EUCLIDEAN,
                Overrides.builder().maxDoc(5).build()
            );
            try (FlatVectorsReader reader = newReader(readState)) {
                assertNull(reader.getRandomVectorScorer(FIELD_NAME, new float[] { 1f, 1f, 1f, 1f }));
            }
        }
    }

    @SneakyThrows
    public void testSearch_nonMmapDirectory_collectsAllVectorsWithoutThrowing() {
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] vectors = { { 1.5f, -2.5f, 3.25f, 0.0f }, { -1.0f, 2.0f, -3.0f, 4.0f }, { 0.5f, 0.5f, 0.5f, 0.5f } };
            SegmentReadState readState = writeRawSegment(dir, vectors, VectorSimilarityFunction.EUCLIDEAN);

            try (FlatVectorsReader reader = newReader(readState)) {
                float[] query = { 1f, 1f, 1f, 1f };
                KnnCollector collector = new TopKnnCollector(vectors.length, Integer.MAX_VALUE);
                reader.search(FIELD_NAME, query, collector, AcceptDocs.fromLiveDocs(null, vectors.length));

                // topDocs() drains the underlying queue, so call it exactly once.
                ScoreDoc[] scoreDocs = collector.topDocs().scoreDocs;
                assertEquals(vectors.length, scoreDocs.length);
            }
        }
    }

    @SneakyThrows
    public void testSearch_emptySegment_collectsNothingWithoutThrowing() {
        try (Directory dir = new ByteBuffersDirectory()) {
            SegmentReadState readState = writeRawSegment(
                dir,
                new float[0][0],
                VectorSimilarityFunction.EUCLIDEAN,
                Overrides.builder().maxDoc(5).build()
            );
            try (FlatVectorsReader reader = newReader(readState)) {
                KnnCollector collector = new TopKnnCollector(5, Integer.MAX_VALUE);
                reader.search(FIELD_NAME, new float[] { 1f, 1f, 1f, 1f }, collector, AcceptDocs.fromLiveDocs(null, 5));
                assertEquals(0, collector.topDocs().scoreDocs.length);
            }
        }
    }

    @SneakyThrows
    public void testGetFloatVectorValues_mmapDirectory_wrapsInMMapFloatVectorValuesAndDecodesCorrectly() {
        try (Directory dir = new MMapDirectory(createTempDir())) {
            float[][] vectors = { { 1.5f, -2.5f, 3.25f, 0.0f }, { -1.0f, 2.0f, -3.0f, 4.0f } };
            SegmentReadState readState = writeRawSegment(dir, vectors, VectorSimilarityFunction.EUCLIDEAN);

            try (FlatVectorsReader reader = newReader(readState)) {
                FloatVectorValues values = reader.getFloatVectorValues(FIELD_NAME);
                assertTrue("expected mmap-backed values for this test", values instanceof MMapFloatVectorValues);
                assertEquals(DIMENSION, values.dimension());
                assertEquals(vectors.length, values.size());

                for (int i = 0; i < vectors.length; i++) {
                    float[] actual = values.vectorValue(i);
                    for (int d = 0; d < DIMENSION; d++) {
                        float expected = Float.float16ToFloat(Float.floatToFloat16(vectors[i][d]));
                        assertEquals("vector " + i + " dim " + d, expected, actual[d], 0.0f);
                    }
                }

                // Exercises MMapFloatVectorValues' iterator()/ordToDoc() pass-throughs directly.
                FloatVectorValues.DocIndexIterator iterator = values.iterator();
                int ord = 0;
                for (int doc = iterator.nextDoc(); doc != org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS; doc = iterator
                    .nextDoc()) {
                    assertEquals(ord, doc);
                    assertEquals(ord, values.ordToDoc(ord));
                    ord++;
                }
                assertEquals(vectors.length, ord);
            }
        }
    }

    @SneakyThrows
    public void testGetRandomVectorScorer_mmapDirectory_matchesExpectedSimilarity() {
        try (Directory dir = new MMapDirectory(createTempDir())) {
            float[][] vectors = { { 1.5f, -2.5f, 3.25f, 0.0f }, { -1.0f, 2.0f, -3.0f, 4.0f }, { 0.5f, 0.5f, 0.5f, 0.5f } };
            SegmentReadState readState = writeRawSegment(dir, vectors, VectorSimilarityFunction.EUCLIDEAN);

            try (FlatVectorsReader reader = newReaderWithRealScorer(readState)) {
                float[] query = { 1f, 1f, 1f, 1f };
                RandomVectorScorer scorer = reader.getRandomVectorScorer(FIELD_NAME, query);
                assertNotNull(scorer);

                for (int i = 0; i < vectors.length; i++) {
                    float[] decoded = new float[DIMENSION];
                    for (int d = 0; d < DIMENSION; d++) {
                        decoded[d] = Float.float16ToFloat(Float.floatToFloat16(vectors[i][d]));
                    }
                    float expected = VectorSimilarityFunction.EUCLIDEAN.compare(query, decoded);
                    assertEquals("Vector " + i, expected, scorer.score(i), 1e-3f);
                }
            }
        }
    }

    @SneakyThrows
    public void testSearch_mmapDirectory_collectsAllVectorsWithoutThrowing() {
        try (Directory dir = new MMapDirectory(createTempDir())) {
            float[][] vectors = { { 1.5f, -2.5f, 3.25f, 0.0f }, { -1.0f, 2.0f, -3.0f, 4.0f }, { 0.5f, 0.5f, 0.5f, 0.5f } };
            SegmentReadState readState = writeRawSegment(dir, vectors, VectorSimilarityFunction.EUCLIDEAN);

            try (FlatVectorsReader reader = newReaderWithRealScorer(readState)) {
                KnnCollector collector = new TopKnnCollector(vectors.length, Integer.MAX_VALUE);
                reader.search(FIELD_NAME, new float[] { 1f, 1f, 1f, 1f }, collector, AcceptDocs.fromLiveDocs(null, vectors.length));
                assertEquals(vectors.length, collector.topDocs().scoreDocs.length);
            }
        }
    }

    @SneakyThrows
    public void testGetRandomVectorScorer_cosineSimilarity_usesPureJavaFallbackAndMatchesExpected() {
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] vectors = { { 1.5f, -2.5f, 3.25f, 0.0f }, { -1.0f, 2.0f, -3.0f, 4.0f } };
            SegmentReadState readState = writeRawSegment(dir, vectors, VectorSimilarityFunction.COSINE);

            try (FlatVectorsReader reader = newReader(readState)) {
                float[] query = { 1f, 1f, 1f, 1f };
                RandomVectorScorer scorer = reader.getRandomVectorScorer(FIELD_NAME, query);
                assertNotNull(scorer);

                for (int i = 0; i < vectors.length; i++) {
                    float[] decoded = new float[DIMENSION];
                    for (int d = 0; d < DIMENSION; d++) {
                        decoded[d] = Float.float16ToFloat(Float.floatToFloat16(vectors[i][d]));
                    }
                    float expected = VectorSimilarityFunction.COSINE.compare(query, decoded);
                    assertEquals("Vector " + i, expected, scorer.score(i), 1e-3f);
                }
            }
        }
    }

    @SneakyThrows
    public void testGetFloatVectorValues_repeatedOrdinal_hitsCacheAndReturnsSameArray() {
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] vectors = { { 1.5f, -2.5f, 3.25f, 0.0f }, { -1.0f, 2.0f, -3.0f, 4.0f } };
            SegmentReadState readState = writeRawSegment(dir, vectors, VectorSimilarityFunction.EUCLIDEAN);

            try (FlatVectorsReader reader = newReader(readState)) {
                FloatVectorValues values = reader.getFloatVectorValues(FIELD_NAME);
                float[] first = values.vectorValue(0);
                float[] second = values.vectorValue(0);
                assertSame("repeated access to the same ordinal should hit the cache", first, second);
            }
        }
    }

    @SneakyThrows
    public void testOpenDataInput_truncatedVectorDataFile_throws() {
        try (Directory dir = new ByteBuffersDirectory()) {
            SegmentReadState readState = writeRawSegment(dir, new float[][] { { 1f, 2f, 3f, 4f } }, VectorSimilarityFunction.EUCLIDEAN);
            String vectorDataFileName = IndexFileNames.segmentFileName("_0", "", KNN1040HalfFloatFlatVectorsFormat.VECTOR_DATA_EXTENSION);
            dir.deleteFile(vectorDataFileName);
            try (IndexOutput truncated = dir.createOutput(vectorDataFileName, IOContext.DEFAULT)) {
                truncated.writeByte((byte) 0);
            }
            expectThrows(Exception.class, () -> newReader(readState));
        }
    }

    @SneakyThrows
    public void testSearch_collectorRequestsZeroResults_returnsImmediatelyWithoutCollecting() {
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] vectors = { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f } };
            SegmentReadState readState = writeRawSegment(dir, vectors, VectorSimilarityFunction.EUCLIDEAN);

            try (FlatVectorsReader reader = newReader(readState)) {
                KnnCollector mockCollector = Mockito.mock(KnnCollector.class);
                Mockito.when(mockCollector.k()).thenReturn(0);

                reader.search(FIELD_NAME, new float[] { 1f, 1f, 1f, 1f }, mockCollector, AcceptDocs.fromLiveDocs(null, vectors.length));

                Mockito.verify(mockCollector, Mockito.never()).collect(Mockito.anyInt(), Mockito.anyFloat());
            }
        }
    }

    @SneakyThrows
    public void testSearch_earlyTerminationMidLoop_stopsCollectingRemainingVectors() {
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] vectors = { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f }, { 9f, 10f, 11f, 12f } };
            SegmentReadState readState = writeRawSegment(dir, vectors, VectorSimilarityFunction.EUCLIDEAN);

            try (FlatVectorsReader reader = newReader(readState)) {
                KnnCollector mockCollector = Mockito.mock(KnnCollector.class);
                Mockito.when(mockCollector.k()).thenReturn(5);
                // False for the first vector (loop proceeds), true from then on (loop breaks).
                Mockito.when(mockCollector.earlyTerminated()).thenReturn(false, true);

                reader.search(FIELD_NAME, new float[] { 1f, 1f, 1f, 1f }, mockCollector, AcceptDocs.fromLiveDocs(null, vectors.length));

                Mockito.verify(mockCollector, Mockito.atMost(1)).incVisitedCount(Mockito.anyInt());
            }
        }
    }

    @SneakyThrows
    public void testSearch_moreVectorsThanBulkBatchSize_flushesFullBatchMidLoop() {
        try (Directory dir = new ByteBuffersDirectory()) {
            int numVectors = 70; // exceeds the reader's internal BULK_SCORE_BATCH_SIZE of 64
            float[][] vectors = new float[numVectors][DIMENSION];
            for (int i = 0; i < numVectors; i++) {
                for (int d = 0; d < DIMENSION; d++) {
                    vectors[i][d] = i + d;
                }
            }
            SegmentReadState readState = writeRawSegment(dir, vectors, VectorSimilarityFunction.EUCLIDEAN);

            try (FlatVectorsReader reader = newReader(readState)) {
                KnnCollector collector = new TopKnnCollector(numVectors, Integer.MAX_VALUE);
                reader.search(FIELD_NAME, new float[] { 1f, 1f, 1f, 1f }, collector, AcceptDocs.fromLiveDocs(null, numVectors));

                assertEquals(numVectors, collector.topDocs().scoreDocs.length);
            }
        }
    }

    @SneakyThrows
    public void testGetFlatVectorScorer_returnsInjectedScorerInstance() {
        try (Directory dir = new ByteBuffersDirectory()) {
            SegmentReadState readState = writeRawSegment(dir, new float[][] { { 1f, 2f, 3f, 4f } }, VectorSimilarityFunction.EUCLIDEAN);
            FlatVectorsScorer scorer = Mockito.mock(FlatVectorsScorer.class);
            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsReader(readState, scorer)) {
                assertSame(scorer, reader.getFlatVectorScorer(FIELD_NAME));
            }
        }
    }

    private FlatVectorsReader newReader(SegmentReadState readState) throws Exception {
        FlatVectorsScorer scorer = Mockito.mock(FlatVectorsScorer.class);
        return new KNN1040HalfFloatFlatVectorsReader(readState, scorer);
    }

    /**
     * Like {@link #newReader}, but with the real {@link KNN1040HalfFloatVectorScorer} chain instead of
     * a mock. {@code selectScorer}'s mmap-native branch calls {@code flatVectorsScorer.getRandomVectorScorer}
     * directly (unlike the non-mmap fallback branch, which never touches the injected scorer), so a
     * mock would return null there instead of exercising real scoring.
     */
    private FlatVectorsReader newReaderWithRealScorer(SegmentReadState readState) throws Exception {
        FlatVectorsScorer scorer = new KNN1040HalfFloatVectorScorer(
            new PrefetchableFlatVectorScorer(new NativeEngines990KnnVectorsScorer(FlatVectorsScorerProvider.getLucene99FlatVectorsScorer()))
        );
        return new KNN1040HalfFloatFlatVectorsReader(readState, scorer);
    }

    private static final class Overrides {
        private final Integer metaDimension;
        private final Integer metaSimilarityOrdinal;
        private final Integer metaEncodingOrdinal;
        private final Long metaVectorDataLength;
        private final Integer metaFieldNumber;
        private final int[] docIds;
        private final Integer maxDoc;

        private Overrides(
            Integer metaDimension,
            Integer metaSimilarityOrdinal,
            Integer metaEncodingOrdinal,
            Long metaVectorDataLength,
            Integer metaFieldNumber,
            int[] docIds,
            Integer maxDoc
        ) {
            this.metaDimension = metaDimension;
            this.metaSimilarityOrdinal = metaSimilarityOrdinal;
            this.metaEncodingOrdinal = metaEncodingOrdinal;
            this.metaVectorDataLength = metaVectorDataLength;
            this.metaFieldNumber = metaFieldNumber;
            this.docIds = docIds;
            this.maxDoc = maxDoc;
        }

        static Builder builder() {
            return new Builder();
        }

        static final class Builder {
            private Integer metaDimension;
            private Integer metaSimilarityOrdinal;
            private Integer metaEncodingOrdinal;
            private Long metaVectorDataLength;
            private Integer metaFieldNumber;
            private int[] docIds;
            private Integer maxDoc;

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

            Builder metaFieldNumber(int v) {
                this.metaFieldNumber = v;
                return this;
            }

            Builder docIds(int[] v) {
                this.docIds = v;
                return this;
            }

            Builder maxDoc(int v) {
                this.maxDoc = v;
                return this;
            }

            Overrides build() {
                return new Overrides(
                    metaDimension,
                    metaSimilarityOrdinal,
                    metaEncodingOrdinal,
                    metaVectorDataLength,
                    metaFieldNumber,
                    docIds,
                    maxDoc
                );
            }
        }
    }

    private SegmentReadState writeRawSegment(Directory dir, float[][] vectors, VectorSimilarityFunction similarity) throws Exception {
        return writeRawSegment(dir, vectors, similarity, Overrides.builder().build());
    }

    private SegmentReadState writeRawSegment(Directory dir, float[][] vectors, VectorSimilarityFunction similarity, Overrides overrides)
        throws Exception {
        FieldInfo fieldInfo = createFieldInfo(similarity);
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });
        int maxDoc = overrides.maxDoc != null ? overrides.maxDoc : vectors.length;
        SegmentInfo segmentInfo = new SegmentInfo(
            dir,
            Version.LATEST,
            Version.LATEST,
            "_0",
            maxDoc,
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
            for (int ord = 0; ord < vectors.length; ord++) {
                KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE.floatToByteArray(vectors[ord], outputBuffer, DIMENSION);
                vectorData.writeBytes(outputBuffer, 0, outputBuffer.length);
                docsWithField.add(overrides.docIds != null ? overrides.docIds[ord] : ord);
            }
            long vectorDataLength = vectorData.getFilePointer() - vectorDataOffset;

            meta.writeInt(overrides.metaFieldNumber != null ? overrides.metaFieldNumber : fieldInfo.number);
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
                maxDoc,
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
