/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.scorer.PrefetchableFlatVectorScorer.PrefetchableRandomVectorScorer;
import org.opensearch.knn.jni.SimdFp16;

import static org.mockito.Mockito.mock;

public class KNN1040HalfFloatFlatVectorsValuesTests extends KNNTestCase {

    @SneakyThrows
    public void testSelectFallbackScorer_nativeTierAvailable_isWrappedInPrefetchableScorer() {
        assertFallbackScorerIsPrefetchable(VectorSimilarityFunction.EUCLIDEAN, true);
    }

    @SneakyThrows
    public void testSelectFallbackScorer_noNativeKernel_isStillWrappedInPrefetchableScorer() {
        // COSINE never reaches the native branch at all (see NativeEngines990KnnVectorsScorer#getNativeFunctionType),
        // so this exercises the plain-Java anonymous scorer branch instead.
        assertFallbackScorerIsPrefetchable(VectorSimilarityFunction.COSINE, true);
    }

    @SneakyThrows
    private void assertFallbackScorerIsPrefetchable(VectorSimilarityFunction similarityFunction, boolean simdSupported) {
        final KNN1040HalfFloatFlatVectorsValues mockValues = mock(KNN1040HalfFloatFlatVectorsValues.class);
        final float[] target = new float[] { 1f, 2f, 3f };

        try (MockedStatic<SimdFp16> mockedSimdFp16 = Mockito.mockStatic(SimdFp16.class)) {
            mockedSimdFp16.when(SimdFp16::isSIMDSupported).thenReturn(simdSupported);
            RandomVectorScorer scorer = KNN1040HalfFloatFlatVectorsValues.selectFallbackScorer(mockValues, target, similarityFunction);
            assertTrue(scorer instanceof PrefetchableRandomVectorScorer);
        }
    }
}
