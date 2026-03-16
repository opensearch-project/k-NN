/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorScorer;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat.ScalarEncoding;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsReader;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsWriter;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.junit.Test;
import org.opensearch.knn.KNNTestCase;

import java.lang.reflect.Field;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Tests that the native SIMD BBQ scorer ({@link Faiss104ScalerQuantizedVectorScorer}) produces scores
 * matching Lucene's {@link Lucene104ScalarQuantizedVectorScorer} (the source of truth).
 * <p>
 * Uses the Lucene codec pipeline directly:
 * 1. {@link Lucene104ScalarQuantizedVectorsFormat#fieldsWriter} to quantize and write vectors.
 * 2. {@link Lucene104ScalarQuantizedVectorsReader} with {@link Lucene104ScalarQuantizedVectorScorer} → truth.
 * 3. {@link Lucene104ScalarQuantizedVectorsReader} with {@link Faiss104ScalerQuantizedVectorScorer} → test subject.
 * 4. Compare scores.
 */
public class FaissBBQBulkSimdScorerTests extends KNNTestCase {

    private static final String FIELD_NAME = "vector";
    private static final int NUM_VECTORS = 500;

    @Test
    public void testBBQEuclideanScoring() {
        // TODO : Remove this once generic version of bulk simd is added
        // for (int dim : Arrays.asList(1, 7, 77, 128, 512, 777, 1024, 10240, 30000, 65535)) {
        // System.out.println("Dimension=" + dim);
        // doTest(VectorSimilarityFunction.EUCLIDEAN, dim);
        // }
    }

    @Test
    public void testBBQMaxInnerProductScoring() {
        // TODO : Remove this once generic version of bulk simd is added
        // for (int dim : Arrays.asList(1, 7, 77, 128, 512, 777, 1024, 10240, 30000, 65535)) {
        // System.out.println("Dimension=" + dim);
        // doTest(VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, dim);
        // }
    }

    @SneakyThrows
    private void doTest(final VectorSimilarityFunction similarityFunction, final int dimension) {
        final ScalarEncoding encoding = ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE;
        final FlatVectorsScorer defaultScorer = FlatVectorScorerUtil.getLucene99FlatVectorsScorer();
        final FlatVectorsFormat rawVectorFormat = getRawVectorFormat();

        // Build FieldInfo for our vector field
        final FieldInfo fieldInfo = new FieldInfo(
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
            dimension,
            VectorEncoding.FLOAT32,
            similarityFunction,
            false,
            false
        );
        final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });

        final java.nio.file.Path tempDir = createTempDir();
        try (MMapDirectory dir = new MMapDirectory(tempDir)) {
            // Build SegmentInfo
            final SegmentInfo segmentInfo = new SegmentInfo(
                dir,
                Version.LATEST,
                Version.LATEST,
                "_0",
                NUM_VECTORS,
                false,
                false,
                null,
                Collections.emptyMap(),
                StringHelper.randomId(),
                new HashMap<>(),
                null
            );

            final SegmentWriteState writeState = new SegmentWriteState(
                InfoStream.NO_OUTPUT,
                dir,
                segmentInfo,
                fieldInfos,
                null,
                IOContext.DEFAULT
            );

            // ---- Step 1: Write vectors using Lucene104ScalarQuantizedVectorsWriter ----
            final Lucene104ScalarQuantizedVectorScorer luceneVectorScorer = new Lucene104ScalarQuantizedVectorScorer(defaultScorer);

            try (
                FlatVectorsWriter writer = new Lucene104ScalarQuantizedVectorsWriter(
                    writeState,
                    encoding,
                    rawVectorFormat.fieldsWriter(writeState),
                    luceneVectorScorer
                )
            ) {
                @SuppressWarnings("unchecked")
                FlatFieldVectorsWriter<float[]> fieldWriter = (FlatFieldVectorsWriter<float[]>) writer.addField(fieldInfo);

                for (int i = 0; i < NUM_VECTORS; i++) {
                    fieldWriter.addValue(i, randomVector(dimension));
                }

                writer.flush(NUM_VECTORS, null);
                writer.finish();
            }

            final SegmentReadState readState = new SegmentReadState(dir, segmentInfo, fieldInfos, IOContext.DEFAULT);

            // ---- Step 2: Lucene scorer (source of truth) ----
            final float[] queryVector = randomVector(dimension);
            final RandomVectorScorer truthScorer;
            try (
                FlatVectorsReader truthReader = new Lucene104ScalarQuantizedVectorsReader(
                    readState,
                    rawVectorFormat.fieldsReader(readState),
                    luceneVectorScorer
                )
            ) {
                truthScorer = truthReader.getRandomVectorScorer(FIELD_NAME, queryVector);
                assertNotNull("Truth scorer should not be null", truthScorer);

                // ---- Step 3: SIMD scorer (test subject) ----
                final Faiss104ScalerQuantizedVectorScorer simdFlatScorer = new Faiss104ScalerQuantizedVectorScorer(defaultScorer);

                try (
                    FlatVectorsReader testReader = new Lucene104ScalarQuantizedVectorsReader(
                        readState,
                        rawVectorFormat.fieldsReader(readState),
                        simdFlatScorer
                    )
                ) {
                    RandomVectorScorer testScorer = testReader.getRandomVectorScorer(FIELD_NAME, queryVector);
                    assertNotNull("Test scorer should not be null", testScorer);

                    // ---- Step 4: Compare scores ----
                    int maxOrd = truthScorer.maxOrd();
                    assertEquals("maxOrd mismatch", maxOrd, testScorer.maxOrd());

                    for (int ord = 0; ord < maxOrd; ord++) {
                        float actual = testScorer.score(ord);
                        float expected = truthScorer.score(ord);
                        assertEquals("Score mismatch at ord=" + ord + " for " + similarityFunction, expected, actual, 1e-2);
                    }

                    // Bulk scoring with various batch sizes to exercise batch-of-8, batch-of-4, and tail
                    for (int batchSize : new int[] { 1, 3, 4, 5, 7, 8, 10, 21, 44 }) {
                        if (batchSize > maxOrd) {
                            continue;
                        }
                        int[] ords = new int[batchSize];
                        float[] bulkScores = new float[batchSize];
                        for (int j = 0; j < batchSize; j++) {
                            ords[j] = ThreadLocalRandom.current().nextInt(NUM_VECTORS);
                        }
                        testScorer.bulkScore(ords, bulkScores, batchSize);
                        for (int j = 0; j < batchSize; j++) {
                            float expected = truthScorer.score(ords[j]);
                            assertEquals(
                                "Bulk score mismatch at ord=" + ords[j] + " (batch=" + batchSize + ") for " + similarityFunction,
                                expected,
                                bulkScores[j],
                                1e-2
                            );
                        }
                    }
                }
            }
        }

        // Clean up temp directory to avoid running out of disk space across 65k iterations
        IOUtils.rm(tempDir);
    }

    /**
     * Extracts the private static {@code rawVectorFormat} field from
     * {@link Lucene104ScalarQuantizedVectorsFormat} via reflection.
     */
    private static FlatVectorsFormat getRawVectorFormat() throws Exception {
        Field field = Lucene104ScalarQuantizedVectorsFormat.class.getDeclaredField("rawVectorFormat");
        field.setAccessible(true);
        return (FlatVectorsFormat) field.get(null);
    }

    private static float[] randomVector(int dimension) {
        float[] v = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            v[i] = ThreadLocalRandom.current().nextFloat() * 2 - 1;
        }
        return v;
    }
}
