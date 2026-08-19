/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.opensearch.common.Randomness;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorScorer;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;
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
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.Version;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.junit.Test;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNN1040Codec.KNN1040ScalarQuantizedVectorScorer;

import java.lang.reflect.Field;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Tests that the native SIMD SQ scorer ({@link KNN1040ScalarQuantizedVectorScorer}) produces scores
 * matching Lucene's {@link Lucene104ScalarQuantizedVectorScorer} (the source of truth).
 * <p>
 * Uses the Lucene codec pipeline directly:
 * 1. {@link Lucene104ScalarQuantizedVectorsFormat#fieldsWriter} to quantize and write vectors.
 * 2. {@link Lucene104ScalarQuantizedVectorsReader} with {@link Lucene104ScalarQuantizedVectorScorer} → truth.
 * 3. {@link Lucene104ScalarQuantizedVectorsReader} with {@link KNN1040ScalarQuantizedVectorScorer} → test subject.
 * 4. Compare scores.
 */
public class FaissScalarQuantizedBulkSimdScorerTests extends KNNTestCase {

    private static final String FIELD_NAME = "vector";
    private static final int NUM_VECTORS = 500;

    @Test
    public void testSQCosineScoring() {
        for (int dim : Arrays.asList(1, 7, 56, 57, 77, 128, 512, 777, 1024)) {
            System.out.println("Dimension=" + dim);
            doTest(VectorSimilarityFunction.COSINE, dim);
        }
    }

    @Test
    public void testSQEuclideanScoring() {
        for (int dim : Arrays.asList(1, 7, 77, 128, 512, 777, 1024, 10240, 30000, 65535)) {
            System.out.println("Dimension=" + dim);
            doTest(VectorSimilarityFunction.EUCLIDEAN, dim);
        }
    }

    @Test
    public void testSQMaxInnerProductScoring() {
        for (int dim : Arrays.asList(1, 7, 77, 128, 512, 777, 1024, 10240, 30000, 65535)) {
            System.out.println("Dimension=" + dim);
            doTest(VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, dim);
        }
    }

    /**
     * The SQ_COSINE kernel treats cosine as a plain inner product ((1 + dot) / 2), which is only
     * correct when both the stored vectors and the query are L2-normalized. getRandomVectorScorer()
     * guards the query half of that invariant with an assertion; verify that an un-normalized cosine
     * query trips it. Assertions are enabled in the test JVM (-ea); guard against environments where
     * they are not so the test does not spuriously fail.
     */
    @Test
    @SneakyThrows
    public void testSQCosine_whenQueryNotNormalized_thenAssertionError() {
        boolean assertionsEnabled = false;
        assert assertionsEnabled = true; // intentional side effect
        assumeTrue("Assertions must be enabled (-ea) to exercise the normalization guard", assertionsEnabled);

        final int dimension = 128;
        final ScalarEncoding encoding = ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE;
        final FlatVectorsScorer defaultScorer = FlatVectorScorerUtil.getLucene99FlatVectorsScorer();
        final FlatVectorsFormat rawVectorFormat = getRawVectorFormat();

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
            VectorSimilarityFunction.COSINE,
            false,
            false
        );
        final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });

        final Path tempDir = createTempDir();
        try (MMapDirectory dir = new MMapDirectory(tempDir)) {
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
                    float[] vec = randomCosineVector(dimension);
                    VectorUtil.l2normalize(vec);
                    fieldWriter.addValue(i, vec);
                }
                writer.flush(NUM_VECTORS, null);
                writer.finish();
            }

            final SegmentReadState readState = new SegmentReadState(dir, segmentInfo, fieldInfos, IOContext.DEFAULT);
            try (
                FlatVectorsReader reader = new Lucene104ScalarQuantizedVectorsReader(
                    readState,
                    rawVectorFormat.fieldsReader(readState),
                    luceneVectorScorer
                )
            ) {
                final KNN1040ScalarQuantizedVectorScorer simdFlatScorer = new KNN1040ScalarQuantizedVectorScorer(defaultScorer);

                // Deliberately un-normalized query: shift every component so the L2 norm is far from 1.
                final float[] unnormalizedQuery = randomCosineVector(dimension);
                for (int i = 0; i < dimension; i++) {
                    unnormalizedQuery[i] += 10.0f;
                }
                assertTrue(
                    "Test query must be un-normalized",
                    Math.abs(VectorUtil.dotProduct(unnormalizedQuery, unnormalizedQuery) - 1.0f) > 1e-2f
                );

                final var floatVectorValues = reader.getFloatVectorValues(FIELD_NAME);
                expectThrows(
                    AssertionError.class,
                    () -> simdFlatScorer.getRandomVectorScorer(VectorSimilarityFunction.COSINE, floatVectorValues, unnormalizedQuery)
                );
            }
        }

        IOUtils.rm(tempDir);
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

        final Path tempDir = createTempDir();
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
                    float[] vec = similarityFunction == VectorSimilarityFunction.COSINE
                        ? randomCosineVector(dimension)
                        : randomVector(dimension);
                    if (similarityFunction == VectorSimilarityFunction.COSINE) {
                        VectorUtil.l2normalize(vec);
                    }
                    fieldWriter.addValue(i, vec);
                }

                writer.flush(NUM_VECTORS, null);
                writer.finish();
            }

            final SegmentReadState readState = new SegmentReadState(dir, segmentInfo, fieldInfos, IOContext.DEFAULT);

            // ---- Step 2: Lucene scorer (source of truth) ----
            float[] queryVector = similarityFunction == VectorSimilarityFunction.COSINE
                ? randomCosineVector(dimension)
                : randomVector(dimension);
            if (similarityFunction == VectorSimilarityFunction.COSINE) {
                VectorUtil.l2normalize(queryVector);
            }
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
                final KNN1040ScalarQuantizedVectorScorer simdFlatScorer = new KNN1040ScalarQuantizedVectorScorer(defaultScorer);

                RandomVectorScorer testScorer = simdFlatScorer.getRandomVectorScorer(
                    similarityFunction,
                    truthReader.getFloatVectorValues(FIELD_NAME),
                    queryVector
                );
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
                        ords[j] = Randomness.get().nextInt(NUM_VECTORS);
                    }
                    testScorer.bulkScore(ords, bulkScores, batchSize);
                    for (int j = 0; j < batchSize; j++) {
                        float actualBulk = bulkScores[j];
                        float expected = truthScorer.score(ords[j]);
                        assertEquals(
                            "Bulk score mismatch at ord=" + ords[j] + " (batch=" + batchSize + ") for " + similarityFunction,
                            expected,
                            actualBulk,
                            1e-2
                        );
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
            v[i] = Randomness.get().nextFloat() * 2 - 1;
        }
        return v;
    }

    /**
     * Generates a random vector with sufficient variance to avoid zero-length vectors
     * after centroid centering during cosine quantization.
     */
    private static float[] randomCosineVector(int dimension) {
        float[] v = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            v[i] = Randomness.get().nextFloat() * 2 - 1;
        }
        // Ensure the vector has meaningful magnitude by setting a component based on index
        v[0] = Randomness.get().nextFloat() * 0.5f + 0.5f;
        if (dimension > 1) {
            v[dimension - 1] = -(Randomness.get().nextFloat() * 0.5f + 0.5f);
        }
        return v;
    }
}
