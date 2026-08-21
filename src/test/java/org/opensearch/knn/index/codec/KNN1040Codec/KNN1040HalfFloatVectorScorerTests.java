/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.apache.lucene.util.hnsw.UpdateableRandomVectorScorer;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.scorer.PrefetchableFlatVectorScorer.PrefetchableRandomVectorScorer;
import org.opensearch.knn.jni.SimdFp16;
import org.opensearch.knn.memoryoptsearch.faiss.MMapFloatVectorValues;

import java.lang.reflect.Field;

import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Verifies {@link KNN1040HalfFloatVectorScorer#getRandomVectorScorerSupplier} picks the decode-free
 * native SIMD tier when a native FP16 kernel exists for the similarity function and SIMD is
 * available, and otherwise falls back to {@link org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer}
 * - the same crash this class was written to avoid (see class javadoc). This guards against silently
 * regressing back to the slow/crashing fallback for spaces that should be using native SIMD.
 */
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
        // No native FP16 kernel exists for COSINE at all - must fall back even when SIMD is available.
        assertFallbackTierUsed(VectorSimilarityFunction.COSINE, true);
        assertFallbackTierUsed(VectorSimilarityFunction.COSINE, false);
    }

    @SneakyThrows
    public void testGetRandomVectorScorerSupplier_euclideanWithoutSimdSupported_usesFallbackTier() {
        assertFallbackTierUsed(VectorSimilarityFunction.EUCLIDEAN, false);
    }

    // Deliberately never invokes .scorer()/.score(): SimdVectorComputeService's methods are declared
    // `native`, and Mockito's mockStatic does not reliably intercept JNI-linked native methods in this
    // environment - a version of this test that called .score() with a fake address crashed the JVM
    // with a real SIGSEGV inside the native FP16 SIMD library instead of hitting the mock. Verifying
    // via reflection that the address is threaded into the supplier (rather than discarded, as before
    // this change) covers the actual behavior change without touching native code at all.
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

    // Same native-call safety note as above: setScoringOrdinal constructs a real HalfFloatRandomVectorScorer
    // (a legitimate saveSearchContext call with a real target and an empty address array, matching the
    // byte-copy tier's normal production behavior), but this test never calls .score()/.bulkScore(), so
    // it never risks dereferencing anything.
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
