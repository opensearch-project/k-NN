/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.DocAndFloatFeatureBuffer;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.memoryoptsearch.faiss.MMapFloatVectorValues;

import org.apache.lucene.index.Sorter;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class KNN1040HalfFloatFlatVectorsFormatTests extends KNNTestCase {

    private static final String FIELD_NAME = "fp16_vector";
    private static final int DIMENSION = 8;
    private static final int NUM_VECTORS = 10;

    @SneakyThrows
    public void testFormatName() {
        KNN1040HalfFloatFlatVectorsFormat format = new KNN1040HalfFloatFlatVectorsFormat();
        assertEquals("KNN1040HalfFloatFlatVectorsFormat", format.getClass().getSimpleName());
    }

    @SneakyThrows
    public void testGetMaxDimensions() {
        KNN1040HalfFloatFlatVectorsFormat format = new KNN1040HalfFloatFlatVectorsFormat();
        assertEquals(KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE), format.getMaxDimensions("test-field"));
    }

    @SneakyThrows
    public void testWriteAndRead_roundTrip() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            float[][] vectors = generateVectors(NUM_VECTORS, DIMENSION);
            SegmentReadState readState = writeVectors(dir, vectors);

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsFormat().fieldsReader(readState)) {
                FloatVectorValues values = reader.getFloatVectorValues(FIELD_NAME);
                assertNotNull(values);
                assertEquals(DIMENSION, values.dimension());
                assertEquals(NUM_VECTORS, values.size());

                for (int i = 0; i < NUM_VECTORS; i++) {
                    float[] actual = values.vectorValue(i);
                    assertNotNull(actual);
                    assertEquals(DIMENSION, actual.length);
                    for (int d = 0; d < DIMENSION; d++) {
                        float expected = Float.float16ToFloat(Float.floatToFloat16(vectors[i][d]));
                        assertEquals("Vector " + i + " dim " + d, expected, actual[d], 0.0f);
                    }
                }
            }
        }
    }

    @SneakyThrows
    public void testWriteAndRead_cachedVectorValue() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            float[][] vectors = generateVectors(5, DIMENSION);
            SegmentReadState readState = writeVectors(dir, vectors);

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsFormat().fieldsReader(readState)) {
                FloatVectorValues values = reader.getFloatVectorValues(FIELD_NAME);
                float[] first = values.vectorValue(2);
                float[] second = values.vectorValue(2);
                assertSame(first, second);
            }
        }
    }

    @SneakyThrows
    public void testWriteAndRead_vectorByteLength() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            float[][] vectors = generateVectors(3, DIMENSION);
            SegmentReadState readState = writeVectors(dir, vectors);

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsFormat().fieldsReader(readState)) {
                FloatVectorValues values = reader.getFloatVectorValues(FIELD_NAME);
                assertEquals(DIMENSION * Short.BYTES, values.getVectorByteLength());
            }
        }
    }

    @SneakyThrows
    public void testGetByteVectorValues_throws() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            float[][] vectors = generateVectors(3, DIMENSION);
            SegmentReadState readState = writeVectors(dir, vectors);

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsFormat().fieldsReader(readState)) {
                expectThrows(UnsupportedOperationException.class, () -> reader.getByteVectorValues(FIELD_NAME));
            }
        }
    }

    @SneakyThrows
    public void testGetFloatVectorValues_unknownField_throws() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            float[][] vectors = generateVectors(3, DIMENSION);
            SegmentReadState readState = writeVectors(dir, vectors);

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsFormat().fieldsReader(readState)) {
                IllegalArgumentException e = expectThrows(IllegalArgumentException.class, () -> reader.getFloatVectorValues("nonexistent"));
                assertTrue(e.getMessage().contains("nonexistent"));
            }
        }
    }

    private SegmentReadState writeVectors(Directory dir, float[][] vectors) throws Exception {
        return writeVectors(dir, vectors, VectorSimilarityFunction.EUCLIDEAN);
    }

    private SegmentReadState writeVectors(Directory dir, float[][] vectors, VectorSimilarityFunction similarity) throws Exception {
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
        SegmentWriteState writeState = new SegmentWriteState(InfoStream.NO_OUTPUT, dir, segmentInfo, fieldInfos, null, IOContext.DEFAULT);

        KNN1040HalfFloatFlatVectorsFormat format = new KNN1040HalfFloatFlatVectorsFormat();
        try (FlatVectorsWriter writer = format.fieldsWriter(writeState)) {
            @SuppressWarnings("unchecked")
            FlatFieldVectorsWriter<float[]> fieldWriter = (FlatFieldVectorsWriter<float[]>) writer.addField(fieldInfo);

            for (int i = 0; i < vectors.length; i++) {
                fieldWriter.addValue(i, vectors[i]);
            }

            writer.flush(vectors.length, null);
            writer.finish();
        }

        return new SegmentReadState(dir, segmentInfo, fieldInfos, IOContext.DEFAULT);
    }

    @SneakyThrows
    public void testWriteWithSortMap_writesReorderedData() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            float[][] vectors = {
                { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f },
                { 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f } };

            FieldInfo fieldInfo = createFieldInfo();
            FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });

            SegmentInfo segmentInfo = new SegmentInfo(
                dir,
                Version.LATEST,
                Version.LATEST,
                "_0",
                2,
                false,
                false,
                null,
                Collections.emptyMap(),
                StringHelper.randomId(),
                new HashMap<>(),
                null
            );
            SegmentWriteState writeState = new SegmentWriteState(
                InfoStream.NO_OUTPUT,
                dir,
                segmentInfo,
                fieldInfos,
                null,
                IOContext.DEFAULT
            );

            Sorter.DocMap sortMap = new Sorter.DocMap() {
                @Override
                public int oldToNew(int docID) {
                    return docID == 0 ? 1 : 0;
                }

                @Override
                public int newToOld(int docID) {
                    return docID == 0 ? 1 : 0;
                }

                @Override
                public int size() {
                    return 2;
                }
            };

            KNN1040HalfFloatFlatVectorsFormat format = new KNN1040HalfFloatFlatVectorsFormat();
            try (FlatVectorsWriter writer = format.fieldsWriter(writeState)) {
                @SuppressWarnings("unchecked")
                FlatFieldVectorsWriter<float[]> fieldWriter = (FlatFieldVectorsWriter<float[]>) writer.addField(fieldInfo);
                fieldWriter.addValue(0, vectors[0]);
                fieldWriter.addValue(1, vectors[1]);
                writer.flush(2, sortMap);
                writer.finish();
            }

            // Verify writer completes without error and creates files
            SegmentReadState readState = new SegmentReadState(dir, segmentInfo, fieldInfos, IOContext.DEFAULT);
            try (FlatVectorsReader reader = format.fieldsReader(readState)) {
                FloatVectorValues values = reader.getFloatVectorValues(FIELD_NAME);
                assertEquals(2, values.size());
                assertNotNull(values.vectorValue(0));
                assertNotNull(values.vectorValue(1));
            }
        }
    }

    private FieldInfo createFieldInfo() {
        return createFieldInfo(VectorSimilarityFunction.EUCLIDEAN);
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

    @SneakyThrows
    public void testScorer_nonMmapDirectory_matchesExpectedSimilarity() {
        // ByteBuffersDirectory does not back onto mmap'd memory, so getFloatVectorValues() returns
        // the plain (non-MMapFloatVectorValues-wrapped) KNN1040HalfFloatFlatVectorsValues. This uses the
        // scorer() fallback path (used when native mmap addresses are unavailable)
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] vectors = generateVectors(NUM_VECTORS, DIMENSION);
            SegmentReadState readState = writeVectors(dir, vectors);

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsFormat().fieldsReader(readState)) {
                FloatVectorValues values = reader.getFloatVectorValues(FIELD_NAME);
                assertFalse("Expected non-mmap-backed values for this test", values instanceof MMapFloatVectorValues);

                float[] query = generateVectors(1, DIMENSION)[0];
                VectorScorer scorer = values.scorer(query);
                assertNotNull(scorer);

                for (int i = 0; i < NUM_VECTORS; i++) {
                    assertTrue(scorer.iterator().nextDoc() != DocIdSetIterator.NO_MORE_DOCS);
                    float[] decoded = new float[DIMENSION];
                    for (int d = 0; d < DIMENSION; d++) {
                        decoded[d] = Float.float16ToFloat(Float.floatToFloat16(vectors[i][d]));
                    }
                    float expected = VectorSimilarityFunction.EUCLIDEAN.compare(query, decoded);
                    assertEquals("Vector " + i, expected, scorer.score(), 1e-3);
                }
            }
        }
    }

    @SneakyThrows
    public void testScorerBulk_nonMmapDirectory_matchesExpectedSimilarity() {
        // Exercises VectorScorer.bulk(), which is what the reader's exhaustive search() path uses.
        // With mmap unavailable, this goes through KNN1040HalfFloatRandomVectorScorer.bulkScore() when SIMD
        // is supported, or the pure-Java fallback (via the default Bulk implementation) otherwise.
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] vectors = generateVectors(NUM_VECTORS, DIMENSION);
            SegmentReadState readState = writeVectors(dir, vectors);

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsFormat().fieldsReader(readState)) {
                FloatVectorValues values = reader.getFloatVectorValues(FIELD_NAME);
                assertFalse("Expected non-mmap-backed values for this test", values instanceof MMapFloatVectorValues);

                float[] query = generateVectors(1, DIMENSION)[0];
                VectorScorer scorer = values.scorer(query);
                assertNotNull(scorer);

                VectorScorer.Bulk bulk = scorer.bulk(null);
                DocAndFloatFeatureBuffer buffer = new DocAndFloatFeatureBuffer();
                float maxScore = bulk.nextDocsAndScores(NUM_VECTORS, null, buffer);
                assertEquals(NUM_VECTORS, buffer.size);

                float expectedMax = Float.NEGATIVE_INFINITY;
                for (int i = 0; i < NUM_VECTORS; i++) {
                    float[] decoded = new float[DIMENSION];
                    for (int d = 0; d < DIMENSION; d++) {
                        decoded[d] = Float.float16ToFloat(Float.floatToFloat16(vectors[i][d]));
                    }
                    float expected = VectorSimilarityFunction.EUCLIDEAN.compare(query, decoded);
                    assertEquals("Vector " + i, expected, buffer.features[i], 1e-3);
                    expectedMax = Math.max(expectedMax, expected);
                }
                assertEquals(expectedMax, maxScore, 1e-3);
            }
        }
    }

    @SneakyThrows
    public void testFloatVectorValues_implementsHasIndexSlice() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            float[][] vectors = generateVectors(NUM_VECTORS, DIMENSION);
            SegmentReadState readState = writeVectors(dir, vectors);

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsFormat().fieldsReader(readState)) {
                FloatVectorValues values = reader.getFloatVectorValues(FIELD_NAME);
                assertNotNull(values);

                // MMapFloatVectorValues wraps KNN1040HalfFloatFlatVectorsValues which implements HasIndexSlice
                // This enables prefetch in PrefetchableFlatVectorScorer
                assertTrue("FloatVectorValues should implement HasIndexSlice for prefetch support", values instanceof HasIndexSlice);
                HasIndexSlice hasSlice = (HasIndexSlice) values;
                assertNotNull("getSlice() should return non-null IndexInput for mmap-backed values", hasSlice.getSlice());
            }
        }
    }

    /**
     * Verifies {@code Values.scorer(target)} picks the mmap zero-copy path of
     * {@link KNN1040HalfFloatVectorScorer.HalfFloatRandomVectorScorer} via the shared
     * {@code selectScorer}/{@code selectNativeOrJavaScorer} helpers, matching the reader's own
     * selection -- exercised through {@link MMapFloatVectorValues#scorer}, the path Lucene's filtered
     * exact-search fallback and rescore use. Guards against silently falling back to the heap-buffer
     * copy when mmap is available.
     */
    @SneakyThrows
    public void testValuesScorer_mmapDirectory_selectsNativeScorer() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            float[][] vectors = generateVectors(NUM_VECTORS, DIMENSION);
            SegmentReadState readState = writeVectors(dir, vectors, VectorSimilarityFunction.EUCLIDEAN);

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsFormat().fieldsReader(readState)) {
                FloatVectorValues values = reader.getFloatVectorValues(FIELD_NAME);
                assertTrue("expected mmap-backed values for this test", values instanceof MMapFloatVectorValues);

                float[] query = generateVectors(1, DIMENSION)[0];
                VectorScorer vectorScorer = values.scorer(query);
                assertNotNull(vectorScorer);

                RandomVectorScorer innerScorer = unwrapCapturedScorer(vectorScorer);
                RandomVectorScorer.AbstractRandomVectorScorer prefetchDelegate = getPrivateField(
                    innerScorer,
                    "delegate",
                    RandomVectorScorer.AbstractRandomVectorScorer.class
                );
                assertTrue(
                    "scorer() should select HalfFloatRandomVectorScorer when a native similarity type is "
                        + "available, not fall back to the plain-Java scorer",
                    prefetchDelegate instanceof KNN1040HalfFloatVectorScorer.HalfFloatRandomVectorScorer
                );
                Boolean usesMmapAddress = getPrivateField(prefetchDelegate, "usesMmapAddress", Boolean.class);
                assertTrue(
                    "scorer() should use the mmap zero-copy path when mmap and a native similarity type are "
                        + "both available, not fall back to the heap-buffer copy",
                    usesMmapAddress
                );

                DocIdSetIterator iterator = vectorScorer.iterator();
                for (int i = 0; i < NUM_VECTORS; i++) {
                    assertTrue(iterator.nextDoc() != DocIdSetIterator.NO_MORE_DOCS);
                    float[] decoded = new float[DIMENSION];
                    for (int d = 0; d < DIMENSION; d++) {
                        decoded[d] = Float.float16ToFloat(Float.floatToFloat16(vectors[i][d]));
                    }
                    float expected = VectorSimilarityFunction.EUCLIDEAN.compare(query, decoded);
                    assertEquals("Vector " + i, expected, vectorScorer.score(), 1e-3);
                }
            }
        }
    }

    /**
     * Reflectively unwraps the {@code RandomVectorScorer} captured in the anonymous
     * {@link VectorScorer} that {@code scorer()} returns (stored as javac's synthetic
     * {@code val$scorer} field), so production code doesn't need a test-only accessor.
     */
    private static RandomVectorScorer unwrapCapturedScorer(VectorScorer vectorScorer) throws Exception {
        return getPrivateField(vectorScorer, "val$scorer", RandomVectorScorer.class);
    }

    @SuppressWarnings("unchecked")
    private static <T> T getPrivateField(Object target, String fieldName, Class<T> fieldType) throws Exception {
        java.lang.reflect.Field field = target.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        return (T) field.get(target);
    }

    @SneakyThrows
    public void testSearch_cosineWithMmapDirectory_doesNotThrowAndMatchesExpected() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            float[][] vectors = generateVectors(NUM_VECTORS, DIMENSION);
            SegmentReadState readState = writeVectors(dir, vectors, VectorSimilarityFunction.COSINE);

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsFormat().fieldsReader(readState)) {
                assertTrue(
                    "expected mmap-backed values for this test",
                    reader.getFloatVectorValues(FIELD_NAME) instanceof MMapFloatVectorValues
                );

                float[] query = generateVectors(1, DIMENSION)[0];
                KnnCollector collector = new TopKnnCollector(NUM_VECTORS, Integer.MAX_VALUE);
                reader.search(FIELD_NAME, query, collector, AcceptDocs.fromLiveDocs(null, NUM_VECTORS));

                // topDocs() drains the underlying queue, so call it exactly once.
                ScoreDoc[] scoreDocs = collector.topDocs().scoreDocs;
                assertEquals(NUM_VECTORS, scoreDocs.length);
                float expectedBest = bestExpectedScore(vectors, query, VectorSimilarityFunction.COSINE, NUM_VECTORS);
                assertEquals(expectedBest, scoreDocs[0].score, 1e-3);
            }
        }
    }

    @SneakyThrows
    public void testGetRandomVectorScorer_nonMmapDirectory_matchesExpectedSimilarity() {
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] vectors = generateVectors(NUM_VECTORS, DIMENSION);
            SegmentReadState readState = writeVectors(dir, vectors);

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsFormat().fieldsReader(readState)) {
                float[] query = generateVectors(1, DIMENSION)[0];
                RandomVectorScorer scorer = reader.getRandomVectorScorer(FIELD_NAME, query);
                assertNotNull(scorer);
                assertEquals(NUM_VECTORS, scorer.maxOrd());

                for (int i = 0; i < NUM_VECTORS; i++) {
                    float[] decoded = new float[DIMENSION];
                    for (int d = 0; d < DIMENSION; d++) {
                        decoded[d] = Float.float16ToFloat(Float.floatToFloat16(vectors[i][d]));
                    }
                    float expected = VectorSimilarityFunction.EUCLIDEAN.compare(query, decoded);
                    assertEquals("Vector " + i, expected, scorer.score(i), 1e-3);
                }
            }
        }
    }

    @SneakyThrows
    public void testSearch_nonMmapDirectory_collectsAllVectorsWithoutThrowing() {
        // End-to-end exhaustive search() through the same non-mmap fallback path as the test above,
        // confirming the reader's own bulk-scoring loop (not just getRandomVectorScorer()) is safe.
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] vectors = generateVectors(NUM_VECTORS, DIMENSION);
            SegmentReadState readState = writeVectors(dir, vectors);

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsFormat().fieldsReader(readState)) {
                float[] query = generateVectors(1, DIMENSION)[0];
                KnnCollector collector = new TopKnnCollector(NUM_VECTORS, Integer.MAX_VALUE);
                reader.search(FIELD_NAME, query, collector, AcceptDocs.fromLiveDocs(null, NUM_VECTORS));
                assertEquals(NUM_VECTORS, collector.topDocs().scoreDocs.length);
            }
        }
    }

    /**
     * Merges segments under a descending index sort, so later segments land entirely before
     * earlier ones, and checks every doc still has the right vector.
     *
     * <p>A merge that appends {@code docMap.get(doc)} reader-by-reader would emit descending doc
     * ids at the first reader boundary, tripping {@code DocsWithFieldSet}'s strictly-increasing
     * check -- a bug {@code testHalfFloatFlatIndex_forceMerge} can't catch, since without a sort
     * each reader's block is already increasing.
     */
    @SneakyThrows
    public void testMergeWithIndexSort_preservesDocToVectorMapping() {
        final int docsPerSegment = 5;
        final int numSegments = 3;
        final int totalDocs = docsPerSegment * numSegments;
        final String sortFieldName = "sort_key";
        final String idFieldName = "id";

        try (Directory dir = new MMapDirectory(createTempDir())) {
            final Codec codec = new Lucene104Codec() {
                @Override
                public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                    return new KNN1040HalfFloatFlatVectorsFormat();
                }
            };

            final IndexWriterConfig iwc = new IndexWriterConfig().setCodec(codec)
                .setIndexSort(new Sort(new SortField(sortFieldName, SortField.Type.LONG, true)));

            final float[][] vectors = generateVectors(totalDocs, DIMENSION);

            try (IndexWriter writer = new IndexWriter(dir, iwc)) {
                for (int i = 0; i < totalDocs; i++) {
                    Document doc = new Document();
                    doc.add(new KnnFloatVectorField(FIELD_NAME, vectors[i], VectorSimilarityFunction.EUCLIDEAN));
                    doc.add(new NumericDocValuesField(sortFieldName, i));
                    doc.add(new StoredField(idFieldName, i));
                    writer.addDocument(doc);
                    if ((i + 1) % docsPerSegment == 0) {
                        writer.commit();
                    }
                }
                writer.forceMerge(1);
            }

            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                assertEquals("force merge should leave a single segment", 1, reader.leaves().size());
                LeafReader leaf = reader.leaves().get(0).reader();

                FloatVectorValues values = leaf.getFloatVectorValues(FIELD_NAME);
                assertNotNull(values);
                assertEquals(totalDocs, values.size());

                int seen = 0;
                int previousId = Integer.MAX_VALUE;
                KnnVectorValues.DocIndexIterator iterator = values.iterator();
                for (int doc = iterator.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = iterator.nextDoc()) {
                    int id = leaf.storedFields().document(doc).getField(idFieldName).numericValue().intValue();

                    // Descending sort on an ascending key: ids must come back in descending order,
                    // which is what proves the segments actually interleaved during the merge.
                    assertTrue("docs should be in descending id order, got " + id + " after " + previousId, id < previousId);
                    previousId = id;

                    float[] actual = values.vectorValue(iterator.index());
                    for (int d = 0; d < DIMENSION; d++) {
                        float expected = Float.float16ToFloat(Float.floatToFloat16(vectors[id][d]));
                        assertEquals("vector mismatch for id=" + id + " dim=" + d, expected, actual[d], 0.0f);
                    }
                    seen++;
                }
                assertEquals(totalDocs, seen);
            }
        }
    }

    private float bestExpectedScore(float[][] vectors, float[] query, VectorSimilarityFunction similarity, int numVectors) {
        float best = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < numVectors; i++) {
            float[] decoded = new float[DIMENSION];
            for (int d = 0; d < DIMENSION; d++) {
                decoded[d] = Float.float16ToFloat(Float.floatToFloat16(vectors[i][d]));
            }
            best = Math.max(best, similarity.compare(query, decoded));
        }
        return best;
    }

    private float[][] generateVectors(int count, int dimension) {
        float[][] vectors = new float[count][dimension];
        for (int i = 0; i < count; i++) {
            for (int d = 0; d < dimension; d++) {
                vectors[i][d] = (random().nextFloat() * 2 - 1) * 10;
            }
        }
        return vectors;
    }
}
