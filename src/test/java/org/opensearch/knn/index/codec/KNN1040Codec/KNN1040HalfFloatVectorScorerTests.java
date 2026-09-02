/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.apache.lucene.util.hnsw.UpdateableRandomVectorScorer;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.scorer.PrefetchableFlatVectorScorer.PrefetchableRandomVectorScorer;
import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfHalfFloatsSerializer;
import org.opensearch.knn.jni.SimdFp16;
import org.opensearch.knn.jni.SimdVectorComputeService;
import org.opensearch.knn.memoryoptsearch.MemorySegmentAddressExtractorUtil;
import org.opensearch.knn.memoryoptsearch.faiss.MMapFloatVectorValues;

import java.lang.reflect.Field;

import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class KNN1040HalfFloatVectorScorerTests extends KNNTestCase {

    private static final String NATIVE_TIER_CLASS_NAME = "HalfFloatRandomVectorScorerSupplier";

    @SneakyThrows
    public void testGetRandomVectorScorerSupplier_euclideanWithSimdSupported_usesNativeTier() {
        assertNativeTierUsed(VectorSimilarityFunction.EUCLIDEAN, true);
    }

    @SneakyThrows
    public void testGetRandomVectorScorerSupplier_maximumInnerProductWithSimdSupported_usesNativeTier() {
        assertNativeTierUsed(VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, true);
    }

    @SneakyThrows
    public void testGetRandomVectorScorerSupplier_cosine_neverUsesNativeTierRegardlessOfSimd() {
        // No native FP16 kernel exists for COSINE yet so must fall back even when SIMD is available.
        assertFallbackTierUsed(VectorSimilarityFunction.COSINE, true);
        assertFallbackTierUsed(VectorSimilarityFunction.COSINE, false);
    }

    @SneakyThrows
    public void testGetRandomVectorScorerSupplier_euclideanWithoutSimdSupported_usesFallbackTier() {
        assertFallbackTierUsed(VectorSimilarityFunction.EUCLIDEAN, false);
    }

    @SneakyThrows
    public void testGetRandomVectorScorerSupplier_mmapAvailable_threadsAddressIntoSupplier() {
        final FlatVectorsScorer mockDelegate = mock(FlatVectorsScorer.class);
        final KNN1040HalfFloatVectorScorer scorer = new KNN1040HalfFloatVectorScorer(mockDelegate);

        final KNN1040HalfFloatFlatVectorsValues mockValues = mock(KNN1040HalfFloatFlatVectorsValues.class);
        when(mockValues.copy()).thenReturn(mockValues);
        final long[] expectedAddress = new long[] { 123L, 456L };
        final MMapFloatVectorValues mmapValues = new MMapFloatVectorValues(mockValues, expectedAddress);

        try (MockedStatic<SimdFp16> mockedSimdFp16 = Mockito.mockStatic(SimdFp16.class)) {
            mockedSimdFp16.when(SimdFp16::isSIMDSupported).thenReturn(true);

            RandomVectorScorerSupplier supplier = scorer.getRandomVectorScorerSupplier(VectorSimilarityFunction.EUCLIDEAN, mmapValues);
            assertEquals(NATIVE_TIER_CLASS_NAME, supplier.getClass().getSimpleName());

            Field addressField = supplier.getClass().getDeclaredField("addressAndSize");
            addressField.setAccessible(true);
            assertArrayEquals(expectedAddress, (long[]) addressField.get(supplier));
        }
    }

    @SneakyThrows
    public void testGetRandomVectorScorerSupplier_mmapAvailableButSimdUnsupported_stillUsesNativeTier() {
        final FlatVectorsScorer mockDelegate = mock(FlatVectorsScorer.class);
        final KNN1040HalfFloatVectorScorer scorer = new KNN1040HalfFloatVectorScorer(mockDelegate);

        final KNN1040HalfFloatFlatVectorsValues mockValues = mock(KNN1040HalfFloatFlatVectorsValues.class);
        when(mockValues.copy()).thenReturn(mockValues);
        final MMapFloatVectorValues mmapValues = new MMapFloatVectorValues(mockValues, new long[] { 123L, 456L });

        try (MockedStatic<SimdFp16> mockedSimdFp16 = Mockito.mockStatic(SimdFp16.class)) {
            mockedSimdFp16.when(SimdFp16::isSIMDSupported).thenReturn(false);

            RandomVectorScorerSupplier supplier = scorer.getRandomVectorScorerSupplier(VectorSimilarityFunction.EUCLIDEAN, mmapValues);
            assertEquals(NATIVE_TIER_CLASS_NAME, supplier.getClass().getSimpleName());
        }
    }

    @SneakyThrows
    public void testGetRandomVectorScorerSupplier_scorerWrapsDelegateInPrefetchableScorer() {
        final FlatVectorsScorer mockDelegate = mock(FlatVectorsScorer.class);
        final KNN1040HalfFloatVectorScorer scorer = new KNN1040HalfFloatVectorScorer(mockDelegate);

        final KNN1040HalfFloatFlatVectorsValues mockValues = mock(KNN1040HalfFloatFlatVectorsValues.class);
        when(mockValues.copy()).thenReturn(mockValues);
        when(mockValues.vectorValue(anyInt())).thenReturn(new float[] { 1f, 2f, 3f });

        try (MockedStatic<SimdFp16> mockedSimdFp16 = Mockito.mockStatic(SimdFp16.class)) {
            mockedSimdFp16.when(SimdFp16::isSIMDSupported).thenReturn(true);

            RandomVectorScorerSupplier supplier = scorer.getRandomVectorScorerSupplier(VectorSimilarityFunction.EUCLIDEAN, mockValues);
            UpdateableRandomVectorScorer candidateScorer = supplier.scorer();
            candidateScorer.setScoringOrdinal(0);

            Field prefetchableField = candidateScorer.getClass().getDeclaredField("prefetchableDelegate");
            prefetchableField.setAccessible(true);
            assertTrue(prefetchableField.get(candidateScorer) instanceof PrefetchableRandomVectorScorer);
        }
    }

    @SneakyThrows
    public void testGetRandomVectorScorerSupplier_nonHalfFloatValues_delegatesDirectly() {
        final FlatVectorsScorer mockDelegate = mock(FlatVectorsScorer.class);
        final KNN1040HalfFloatVectorScorer scorer = new KNN1040HalfFloatVectorScorer(mockDelegate);
        final FloatVectorValues plainValues = mock(FloatVectorValues.class);
        final RandomVectorScorerSupplier expected = mock(RandomVectorScorerSupplier.class);
        when(mockDelegate.getRandomVectorScorerSupplier(VectorSimilarityFunction.EUCLIDEAN, plainValues)).thenReturn(expected);

        RandomVectorScorerSupplier result = scorer.getRandomVectorScorerSupplier(VectorSimilarityFunction.EUCLIDEAN, plainValues);
        assertSame(expected, result);
    }

    @SneakyThrows
    public void testGetRandomVectorScorer_byteTarget_delegatesToDelegate() {
        final FlatVectorsScorer mockDelegate = mock(FlatVectorsScorer.class);
        final KNN1040HalfFloatVectorScorer scorer = new KNN1040HalfFloatVectorScorer(mockDelegate);
        final KnnVectorValues values = mock(KnnVectorValues.class);
        final byte[] target = { 1, 2, 3 };
        final RandomVectorScorer expected = mock(RandomVectorScorer.class);
        when(mockDelegate.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, values, target)).thenReturn(expected);

        RandomVectorScorer result = scorer.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, values, target);
        assertSame(expected, result);
    }

    @SneakyThrows
    public void testToString_includesDelegateRepresentation() {
        final FlatVectorsScorer mockDelegate = mock(FlatVectorsScorer.class);
        when(mockDelegate.toString()).thenReturn("mockDelegate");
        final KNN1040HalfFloatVectorScorer scorer = new KNN1040HalfFloatVectorScorer(mockDelegate);

        assertEquals("KNN1040HalfFloatVectorScorer(delegate=mockDelegate)", scorer.toString());
    }

    @SneakyThrows
    public void testScorer_requireDelegate_throwsBeforeSetScoringOrdinalIsCalled() {
        RandomVectorScorerSupplier supplier = getRandomVectorScorerSupplier(VectorSimilarityFunction.EUCLIDEAN, true);
        UpdateableRandomVectorScorer candidateScorer = supplier.scorer();

        IllegalStateException e = expectThrows(IllegalStateException.class, () -> candidateScorer.score(0));
        assertTrue(e.getMessage().contains("setScoringOrdinal"));
    }

    @SneakyThrows
    public void testScorer_setScoringOrdinalCalledTwice_reusesDelegateThenScoresAndBulkScores() {
        try (Directory dir = new ByteBuffersDirectory()) {
            int dimension = 4;
            float[][] vectors = { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f }, { 2f, 2f, 2f, 2f } };
            KNN1040HalfFloatFlatVectorsValues values = writeAndOpenValues(dir, "vals.vec", vectors, dimension);

            final FlatVectorsScorer mockDelegate = mock(FlatVectorsScorer.class);
            final KNN1040HalfFloatVectorScorer scorer = new KNN1040HalfFloatVectorScorer(mockDelegate);

            RandomVectorScorerSupplier supplier;
            try (MockedStatic<SimdFp16> mockedSimdFp16 = Mockito.mockStatic(SimdFp16.class)) {
                mockedSimdFp16.when(SimdFp16::isSIMDSupported).thenReturn(true);
                supplier = scorer.getRandomVectorScorerSupplier(VectorSimilarityFunction.EUCLIDEAN, values);
            }
            assertEquals(NATIVE_TIER_CLASS_NAME, supplier.getClass().getSimpleName());

            UpdateableRandomVectorScorer candidateScorer = supplier.scorer();

            candidateScorer.setScoringOrdinal(0);
            candidateScorer.setScoringOrdinal(2);

            float expectedSingle = expectedEuclidean(vectors, dimension, 2, 1);
            assertEquals(expectedSingle, candidateScorer.score(1), 1e-3f);

            float[] scores = new float[2];
            float maxScore = candidateScorer.bulkScore(new int[] { 0, 1 }, scores, 2);
            float expectedMax = Math.max(expectedEuclidean(vectors, dimension, 2, 0), expectedEuclidean(vectors, dimension, 2, 1));
            assertEquals(expectedMax, maxScore, 1e-3f);
        }
    }

    @SneakyThrows
    public void testScorerSupplier_copy_returnsNewSupplierBuiltOverCopiedValues() {
        final FlatVectorsScorer mockDelegate = mock(FlatVectorsScorer.class);
        final KNN1040HalfFloatVectorScorer scorer = new KNN1040HalfFloatVectorScorer(mockDelegate);
        final KNN1040HalfFloatFlatVectorsValues mockValues = mock(KNN1040HalfFloatFlatVectorsValues.class);
        final KNN1040HalfFloatFlatVectorsValues copiedValues = mock(KNN1040HalfFloatFlatVectorsValues.class);
        when(mockValues.copy()).thenReturn(copiedValues);

        try (MockedStatic<SimdFp16> mockedSimdFp16 = Mockito.mockStatic(SimdFp16.class)) {
            mockedSimdFp16.when(SimdFp16::isSIMDSupported).thenReturn(true);

            RandomVectorScorerSupplier supplier = scorer.getRandomVectorScorerSupplier(VectorSimilarityFunction.EUCLIDEAN, mockValues);
            RandomVectorScorerSupplier copiedSupplier = supplier.copy();

            assertEquals(NATIVE_TIER_CLASS_NAME, copiedSupplier.getClass().getSimpleName());
            assertNotSame(supplier, copiedSupplier);
        }
    }

    @SneakyThrows
    public void testHalfFloatRandomVectorScorer_mmapAddress_scoreAndBulkScoreUseNativeAddressPath() {
        try (Directory dir = new MMapDirectory(createTempDir())) {
            int dimension = 4;
            float[][] vectors = { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f } };
            KNN1040HalfFloatFlatVectorsValues values = writeAndOpenValues(dir, "mmap.vec", vectors, dimension);
            long[] addressAndSize = MemorySegmentAddressExtractorUtil.tryExtractAddressAndSize(
                values.getSlice(),
                0,
                values.getSlice().length()
            );
            assertNotNull("test host must support mmap address extraction", addressAndSize);

            float[] target = { 1f, 1f, 1f, 1f };
            KNN1040HalfFloatVectorScorer.HalfFloatRandomVectorScorer halfFloatScorer =
                new KNN1040HalfFloatVectorScorer.HalfFloatRandomVectorScorer(
                    values,
                    target,
                    SimdVectorComputeService.SimilarityFunctionType.FP16_L2,
                    addressAndSize
                );

            float expected0 = expectedEuclidean(target, vectors, dimension, 0);
            float expected1 = expectedEuclidean(target, vectors, dimension, 1);
            assertEquals(expected0, halfFloatScorer.score(0), 1e-3f);

            float[] scores = new float[2];
            float maxScore = halfFloatScorer.bulkScore(new int[] { 0, 1 }, scores, 2);
            assertEquals(Math.max(expected0, expected1), maxScore, 1e-3f);
        }
    }

    @SneakyThrows
    public void testHalfFloatRandomVectorScorer_nonMmap_bulkScoreGrowsBufferAndIdentityIdsPastDefaultBatchSize() {
        try (Directory dir = new ByteBuffersDirectory()) {
            int dimension = 2;
            // Default non-mmap buffer is sized for 64 nodes; exceeding that forces both the byte
            // buffer and the identity-ids array to grow past their initial allocation.
            int numNodes = 100;
            float[][] vectors = new float[numNodes][dimension];
            for (int i = 0; i < numNodes; i++) {
                vectors[i][0] = i;
                vectors[i][1] = i + 1;
            }
            KNN1040HalfFloatFlatVectorsValues values = writeAndOpenValues(dir, "nonmmap.vec", vectors, dimension);

            float[] target = { 1f, 2f };
            KNN1040HalfFloatVectorScorer.HalfFloatRandomVectorScorer halfFloatScorer =
                new KNN1040HalfFloatVectorScorer.HalfFloatRandomVectorScorer(
                    values,
                    target,
                    SimdVectorComputeService.SimilarityFunctionType.FP16_L2,
                    null
                );

            int[] nodes = new int[numNodes];
            for (int i = 0; i < numNodes; i++) {
                nodes[i] = i;
            }
            float[] scores = new float[numNodes];

            float result = halfFloatScorer.bulkScore(nodes, scores, numNodes);
            float expectedMax = 0f;
            for (int i = 0; i < numNodes; i++) {
                expectedMax = Math.max(expectedMax, expectedEuclidean(target, vectors, dimension, i));
            }
            assertEquals(expectedMax, result, 1e-2f);
        }
    }

    private KNN1040HalfFloatFlatVectorsValues writeAndOpenValues(Directory dir, String fileName, float[][] vectors, int dimension)
        throws Exception {
        byte[] buffer = new byte[dimension * Short.BYTES];
        try (IndexOutput out = dir.createOutput(fileName, IOContext.DEFAULT)) {
            for (float[] v : vectors) {
                KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE.floatToByteArray(v, buffer, dimension);
                out.writeBytes(buffer, 0, buffer.length);
            }
        }
        IndexInput in = dir.openInput(fileName, IOContext.DEFAULT);
        return new KNN1040HalfFloatFlatVectorsValues(dimension, vectors.length, in, null, VectorSimilarityFunction.EUCLIDEAN);
    }

    private float expectedEuclidean(float[][] vectors, int dimension, int targetOrd, int candidateOrd) {
        return expectedEuclidean(decodeFp16(vectors[targetOrd], dimension), vectors, dimension, candidateOrd);
    }

    private float expectedEuclidean(float[] target, float[][] vectors, int dimension, int candidateOrd) {
        return VectorSimilarityFunction.EUCLIDEAN.compare(target, decodeFp16(vectors[candidateOrd], dimension));
    }

    private float[] decodeFp16(float[] original, int dimension) {
        float[] decoded = new float[dimension];
        for (int d = 0; d < dimension; d++) {
            decoded[d] = Float.float16ToFloat(Float.floatToFloat16(original[d]));
        }
        return decoded;
    }

    @SneakyThrows
    private void assertNativeTierUsed(VectorSimilarityFunction similarityFunction, boolean simdSupported) {
        RandomVectorScorerSupplier result = getRandomVectorScorerSupplier(similarityFunction, simdSupported);
        assertEquals(NATIVE_TIER_CLASS_NAME, result.getClass().getSimpleName());
    }

    @SneakyThrows
    private void assertFallbackTierUsed(VectorSimilarityFunction similarityFunction, boolean simdSupported) {
        RandomVectorScorerSupplier result = getRandomVectorScorerSupplier(similarityFunction, simdSupported);
        assertNotEquals(NATIVE_TIER_CLASS_NAME, result.getClass().getSimpleName());
    }

    @SneakyThrows
    private RandomVectorScorerSupplier getRandomVectorScorerSupplier(VectorSimilarityFunction similarityFunction, boolean simdSupported) {
        final FlatVectorsScorer mockDelegate = mock(FlatVectorsScorer.class);
        final KNN1040HalfFloatVectorScorer scorer = new KNN1040HalfFloatVectorScorer(mockDelegate);

        final KNN1040HalfFloatFlatVectorsValues mockValues = mock(KNN1040HalfFloatFlatVectorsValues.class);
        when(mockValues.copy()).thenReturn(mockValues);
        // Only reached by the fallback tier's DefaultFlatVectorScorer, which validates the encoding.
        when(mockValues.getEncoding()).thenReturn(VectorEncoding.FLOAT32);

        try (MockedStatic<SimdFp16> mockedSimdFp16 = Mockito.mockStatic(SimdFp16.class)) {
            mockedSimdFp16.when(SimdFp16::isSIMDSupported).thenReturn(simdSupported);
            return scorer.getRandomVectorScorerSupplier(similarityFunction, mockValues);
        }
    }
}
