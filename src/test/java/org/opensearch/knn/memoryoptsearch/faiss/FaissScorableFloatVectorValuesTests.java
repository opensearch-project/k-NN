/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.mockito.ArgumentCaptor;
import org.opensearch.knn.KNNTestCase;

import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class FaissScorableFloatVectorValuesTests extends KNNTestCase {

    private static final int DIMENSION = 4;
    private static final List<float[]> VECTORS = List.of(
        new float[] { 1, 2, 3, 4 },
        new float[] { 5, 6, 7, 8 },
        new float[] { 9, 10, 11, 12 }
    );

    public void testAllFloatVectorValuesMethodsAreOverridden() {
        assertAllMethodsOverridden(FloatVectorValues.class, FaissScorableFloatVectorValues.class);
    }

    public void testNullDelegateThrows() {
        expectThrows(
            IllegalArgumentException.class,
            () -> new FaissScorableFloatVectorValues(null, mock(FlatVectorsScorer.class), VectorSimilarityFunction.EUCLIDEAN, null)
        );
    }

    @SneakyThrows
    public void testDelegatesRandomAccess() {
        final FaissScorableFloatVectorValues wrapper = createWrapper(mock(FlatVectorsScorer.class), null);

        assertEquals(DIMENSION, wrapper.dimension());
        assertEquals(VECTORS.size(), wrapper.size());
        for (int i = 0; i < VECTORS.size(); i++) {
            assertArrayEquals(VECTORS.get(i), wrapper.vectorValue(i), 0.0f);
            assertEquals(i, wrapper.ordToDoc(i));
        }
        assertNull(wrapper.getAcceptOrds(null));
    }

    @SneakyThrows
    public void testScorerReturnsNullForEmptyIndex() {
        final FaissScorableFloatVectorValues wrapper = new FaissScorableFloatVectorValues(
            FloatVectorValues.fromFloats(List.of(), DIMENSION),
            mock(FlatVectorsScorer.class),
            VectorSimilarityFunction.EUCLIDEAN,
            null
        );

        assertNull(wrapper.scorer(new float[] { 1, 2, 3, 4 }));
    }

    @SneakyThrows
    public void testScorerScoresUnwrappedDelegate() {
        // The flat vectors scorer must get the FAISS values themselves, not this wrapper, so it can pick the same scorer it
        // uses for ANN search (for example native SIMD for memory-mapped FP16 values).
        final float[] target = new float[] { 1, 2, 3, 4 };
        final FlatVectorsScorer flatVectorsScorer = mock(FlatVectorsScorer.class);
        when(flatVectorsScorer.getRandomVectorScorer(eq(VectorSimilarityFunction.EUCLIDEAN), any(FloatVectorValues.class), eq(target)))
            .thenReturn(mock(RandomVectorScorer.class));

        createWrapper(flatVectorsScorer, null).scorer(target);

        final ArgumentCaptor<FloatVectorValues> scoredValues = ArgumentCaptor.forClass(FloatVectorValues.class);
        verify(flatVectorsScorer).getRandomVectorScorer(eq(VectorSimilarityFunction.EUCLIDEAN), scoredValues.capture(), eq(target));
        assertFalse(scoredValues.getValue() instanceof FaissScorableFloatVectorValues);
        assertEquals(VECTORS.size(), scoredValues.getValue().size());
    }

    @SneakyThrows
    public void testScorerScoresIteratorOrdinal() {
        final float[] target = new float[] { 1, 2, 3, 4 };
        final RandomVectorScorer rvs = mock(RandomVectorScorer.class);
        when(rvs.score(0)).thenReturn(1.0f);
        when(rvs.score(1)).thenReturn(0.5f);
        final FlatVectorsScorer flatVectorsScorer = mock(FlatVectorsScorer.class);
        when(flatVectorsScorer.getRandomVectorScorer(eq(VectorSimilarityFunction.EUCLIDEAN), any(FloatVectorValues.class), eq(target)))
            .thenReturn(rvs);

        final VectorScorer scorer = createWrapper(flatVectorsScorer, null).scorer(target);
        final DocIdSetIterator iterator = scorer.iterator();

        assertEquals(0, iterator.nextDoc());
        assertEquals(1.0f, scorer.score(), 0.0f);
        assertEquals(1, iterator.nextDoc());
        assertEquals(0.5f, scorer.score(), 0.0f);
    }

    @SneakyThrows
    public void testScorerUsesOverrideIterator() {
        final float[] target = new float[] { 1, 2, 3, 4 };
        final RandomVectorScorer rvs = mock(RandomVectorScorer.class);
        when(rvs.score(2)).thenReturn(0.9f);
        final FlatVectorsScorer flatVectorsScorer = mock(FlatVectorsScorer.class);
        when(flatVectorsScorer.getRandomVectorScorer(eq(VectorSimilarityFunction.EUCLIDEAN), any(FloatVectorValues.class), eq(target)))
            .thenReturn(rvs);
        final KnnVectorValues.DocIndexIterator overrideIterator = mock(KnnVectorValues.DocIndexIterator.class);
        when(overrideIterator.index()).thenReturn(2);

        final FaissScorableFloatVectorValues wrapper = createWrapper(flatVectorsScorer, overrideIterator);
        final VectorScorer scorer = wrapper.scorer(target);

        assertSame(overrideIterator, wrapper.iterator());
        assertSame(overrideIterator, scorer.iterator());
        assertEquals(0.9f, scorer.score(), 0.0f);
    }

    @SneakyThrows
    public void testDelegatesRescorer() {
        final float[] target = new float[] { 1, 2, 3, 4 };
        final VectorScorer expectedScorer = mock(VectorScorer.class);
        final FloatVectorValues delegate = mock(FloatVectorValues.class);
        when(delegate.rescorer(target)).thenReturn(expectedScorer);

        final FaissScorableFloatVectorValues wrapper = new FaissScorableFloatVectorValues(
            delegate,
            mock(FlatVectorsScorer.class),
            VectorSimilarityFunction.EUCLIDEAN,
            null
        );

        assertSame(expectedScorer, wrapper.rescorer(target));
    }

    private static FaissScorableFloatVectorValues createWrapper(
        final FlatVectorsScorer flatVectorsScorer,
        final KnnVectorValues.DocIndexIterator overrideIterator
    ) {
        return new FaissScorableFloatVectorValues(
            FloatVectorValues.fromFloats(VECTORS, DIMENSION),
            flatVectorsScorer,
            VectorSimilarityFunction.EUCLIDEAN,
            overrideIterator
        );
    }
}
