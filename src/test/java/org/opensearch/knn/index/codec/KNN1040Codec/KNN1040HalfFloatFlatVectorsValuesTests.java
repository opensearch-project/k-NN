/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.scorer.PrefetchableFlatVectorScorer.PrefetchableRandomVectorScorer;
import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfHalfFloatsSerializer;
import org.opensearch.knn.jni.SimdFp16;

import static org.mockito.Mockito.mock;

public class KNN1040HalfFloatFlatVectorsValuesTests extends KNNTestCase {

    @SneakyThrows
    public void testSelectNativeOrJavaScorer_nativeTierAvailable_isWrappedInPrefetchableScorer() {
        assertNativeOrJavaScorerIsPrefetchable(VectorSimilarityFunction.EUCLIDEAN, true);
    }

    @SneakyThrows
    public void testSelectNativeOrJavaScorer_noNativeKernel_isStillWrappedInPrefetchableScorer() {
        // COSINE never reaches the native branch at all (see NativeEngines990KnnVectorsScorer#getNativeFunctionType),
        // so this exercises the plain-Java anonymous scorer branch instead.
        assertNativeOrJavaScorerIsPrefetchable(VectorSimilarityFunction.COSINE, true);
    }

    @SneakyThrows
    public void testScorer_zeroVectors_returnsNull() {
        try (Directory dir = new ByteBuffersDirectory()) {
            try (IndexOutput out = dir.createOutput("empty.vec", IOContext.DEFAULT)) {
                // no bytes written; size=0
            }
            try (IndexInput in = dir.openInput("empty.vec", IOContext.DEFAULT)) {
                KNN1040HalfFloatFlatVectorsValues values = new KNN1040HalfFloatFlatVectorsValues(
                    4,
                    0,
                    in,
                    null,
                    VectorSimilarityFunction.EUCLIDEAN
                );
                assertNull(values.scorer(new float[] { 1f, 1f, 1f, 1f }));
            }
        }
    }

    @SneakyThrows
    public void testScorer_nonEmpty_returnsWorkingVectorScorerOverACopiedInstance() {
        try (Directory dir = new ByteBuffersDirectory()) {
            int dimension = 4;
            float[][] vectors = { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f } };
            byte[] buffer = new byte[dimension * Short.BYTES];
            try (IndexOutput out = dir.createOutput("vals.vec", IOContext.DEFAULT)) {
                for (float[] v : vectors) {
                    KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE.floatToByteArray(v, buffer, dimension);
                    out.writeBytes(buffer, 0, buffer.length);
                }
            }

            try (IndexInput in = dir.openInput("vals.vec", IOContext.DEFAULT)) {
                try (MockedStatic<SimdFp16> mockedSimdFp16 = Mockito.mockStatic(SimdFp16.class)) {
                    // Forces the plain-Java fallback branch inside selectNativeOrJavaScorer, avoiding any
                    // dependency on whether the native SIMD lib happens to be loaded in this test run.
                    mockedSimdFp16.when(SimdFp16::isSIMDSupported).thenReturn(false);

                    KNN1040HalfFloatFlatVectorsValues values = new KNN1040HalfFloatFlatVectorsValues(
                        dimension,
                        vectors.length,
                        in,
                        null,
                        VectorSimilarityFunction.EUCLIDEAN
                    );

                    float[] target = { 1f, 1f, 1f, 1f };
                    VectorScorer scorer = values.scorer(target);
                    assertNotNull(scorer);

                    // bulk() just needs to be reachable/return a usable Bulk; the scoring itself is
                    // exercised via score()/iterator() below.
                    assertNotNull(scorer.bulk(DocIdSetIterator.all(vectors.length)));

                    FloatVectorValues.DocIndexIterator iterator = (FloatVectorValues.DocIndexIterator) scorer.iterator();
                    int ord = 0;
                    for (int doc = iterator.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = iterator.nextDoc()) {
                        float[] decoded = new float[dimension];
                        for (int d = 0; d < dimension; d++) {
                            decoded[d] = Float.float16ToFloat(Float.floatToFloat16(vectors[ord][d]));
                        }
                        float expected = VectorSimilarityFunction.EUCLIDEAN.compare(target, decoded);
                        assertEquals("doc " + doc, expected, scorer.score(), 1e-3f);
                        ord++;
                    }
                    assertEquals(vectors.length, ord);
                }
            }
        }
    }

    @SneakyThrows
    private void assertNativeOrJavaScorerIsPrefetchable(VectorSimilarityFunction similarityFunction, boolean simdSupported) {
        final KNN1040HalfFloatFlatVectorsValues mockValues = mock(KNN1040HalfFloatFlatVectorsValues.class);
        final float[] target = new float[] { 1f, 2f, 3f };

        try (MockedStatic<SimdFp16> mockedSimdFp16 = Mockito.mockStatic(SimdFp16.class)) {
            mockedSimdFp16.when(SimdFp16::isSIMDSupported).thenReturn(simdSupported);
            RandomVectorScorer scorer = KNN1040HalfFloatFlatVectorsValues.selectNativeOrJavaScorer(
                mockValues,
                target,
                similarityFunction,
                null
            );
            assertTrue(scorer instanceof PrefetchableRandomVectorScorer);
        }
    }
}
