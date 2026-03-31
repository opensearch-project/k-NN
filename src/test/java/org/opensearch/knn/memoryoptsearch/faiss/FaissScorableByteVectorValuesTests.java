/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocAndFloatFeatureBuffer;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.vectorvalues.TestVectorValues.PreDefinedByteVectorValues;

import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class FaissScorableByteVectorValuesTests extends KNNTestCase {

    private static final int DIMENSION = 4;
    private static final List<byte[]> VECTORS = List.of(new byte[] { 1, 2, 3, 4 }, new byte[] { 5, 6, 7, 8 }, new byte[] { 9, 10, 11, 12 });

    public void testNullDelegateThrows() {
        expectThrows(
            IllegalArgumentException.class,
            () -> new FaissScorableByteVectorValues(null, mock(FlatVectorsScorer.class), VectorSimilarityFunction.EUCLIDEAN, null)
        );
    }

    @SneakyThrows
    public void testDelegatesDimension() {
        final FaissScorableByteVectorValues wrapper = createWrapper();
        assertEquals(DIMENSION, wrapper.dimension());
    }

    @SneakyThrows
    public void testDelegatesSize() {
        final FaissScorableByteVectorValues wrapper = createWrapper();
        assertEquals(VECTORS.size(), wrapper.size());
    }

    @SneakyThrows
    public void testDelegatesVectorValue() {
        final FaissScorableByteVectorValues wrapper = createWrapper();
        for (int i = 0; i < VECTORS.size(); i++) {
            assertArrayEquals(VECTORS.get(i), wrapper.vectorValue(i));
        }
    }

    @SneakyThrows
    public void testDelegatesOrdToDoc() {
        final FaissScorableByteVectorValues wrapper = createWrapper();
        for (int i = 0; i < VECTORS.size(); i++) {
            assertEquals(i, wrapper.ordToDoc(i));
        }
    }

    @SneakyThrows
    public void testDelegatesGetAcceptOrds() {
        final FaissScorableByteVectorValues wrapper = createWrapper();
        assertNull(wrapper.getAcceptOrds(null));
    }

    @SneakyThrows
    public void testDelegatesIterator() {
        final FaissScorableByteVectorValues wrapper = createWrapper();
        final KnnVectorValues.DocIndexIterator iterator = wrapper.iterator();
        assertNotNull(iterator);
        assertEquals(0, iterator.nextDoc());
        assertEquals(0, iterator.index());
        assertEquals(1, iterator.nextDoc());
        assertEquals(1, iterator.index());
        assertEquals(2, iterator.nextDoc());
        assertEquals(2, iterator.index());
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    @SneakyThrows
    public void testCopyReturnsIndependentInstance() {
        final FaissScorableByteVectorValues wrapper = createWrapper();
        final FaissScorableByteVectorValues copy = wrapper.copy();

        assertNotSame(wrapper, copy);
        assertEquals(wrapper.dimension(), copy.dimension());
        assertEquals(wrapper.size(), copy.size());

        // Iterators should be independent
        final KnnVectorValues.DocIndexIterator origIter = wrapper.iterator();
        final KnnVectorValues.DocIndexIterator copyIter = copy.iterator();
        origIter.nextDoc();
        origIter.nextDoc();
        assertEquals(0, copyIter.nextDoc());
    }

    @SneakyThrows
    public void testScorerReturnsNullForEmptyIndex() {
        final FaissScorableByteVectorValues wrapper = new FaissScorableByteVectorValues(
            ByteVectorValues.fromBytes(List.of(), 1),
            mock(FlatVectorsScorer.class),
            VectorSimilarityFunction.EUCLIDEAN,
            null
        );

        assertNull(wrapper.scorer(new byte[] { 1, 2, 3, 4 }));
    }

    @SneakyThrows
    public void testScorerReturnsScorerWithIterator() {
        final float expectedScore = 0.75f;
        final byte[] target = new byte[] { 1, 2, 3, 4 };

        final RandomVectorScorer rvs = mock(RandomVectorScorer.class);
        when(rvs.score(0)).thenReturn(expectedScore);

        final FlatVectorsScorer flatVectorsScorer = mock(FlatVectorsScorer.class);
        when(flatVectorsScorer.getRandomVectorScorer(eq(VectorSimilarityFunction.EUCLIDEAN), any(ByteVectorValues.class), eq(target)))
            .thenReturn(rvs);

        final FaissScorableByteVectorValues wrapper = new FaissScorableByteVectorValues(
            new PreDefinedByteVectorValues(VECTORS),
            flatVectorsScorer,
            VectorSimilarityFunction.EUCLIDEAN,
            null
        );

        final VectorScorer scorer = wrapper.scorer(target);
        assertNotNull(scorer);

        final DocIdSetIterator iterator = scorer.iterator();
        assertNotNull(iterator);

        // Advance to first doc and verify score
        assertEquals(0, iterator.nextDoc());
        assertEquals(expectedScore, scorer.score(), 0.0f);
    }

    @SneakyThrows
    public void testScorerBulkDelegatesBulkScore() {
        final byte[] target = new byte[] { 1, 2, 3, 4 };

        final RandomVectorScorer rvs = mock(RandomVectorScorer.class);
        when(rvs.bulkScore(any(int[].class), any(float[].class), eq(2))).thenReturn(0.9f);

        final FlatVectorsScorer flatVectorsScorer = mock(FlatVectorsScorer.class);
        when(flatVectorsScorer.getRandomVectorScorer(eq(VectorSimilarityFunction.EUCLIDEAN), any(ByteVectorValues.class), eq(target)))
            .thenReturn(rvs);

        final FaissScorableByteVectorValues wrapper = new FaissScorableByteVectorValues(
            new PreDefinedByteVectorValues(VECTORS),
            flatVectorsScorer,
            VectorSimilarityFunction.EUCLIDEAN,
            null
        );

        final VectorScorer scorer = wrapper.scorer(target);
        final VectorScorer.Bulk bulk = scorer.bulk(null);
        assertNotNull(bulk);

        final DocAndFloatFeatureBuffer buffer = new DocAndFloatFeatureBuffer();
        bulk.nextDocsAndScores(Integer.MAX_VALUE, null, buffer);

        verify(rvs).bulkScore(any(int[].class), any(float[].class), eq(VECTORS.size()));
    }

    @SneakyThrows
    public void testScorerUsesIndependentCopy() {
        final byte[] target = new byte[] { 1, 2, 3, 4 };

        final RandomVectorScorer rvs = mock(RandomVectorScorer.class);
        when(rvs.score(0)).thenReturn(1.0f);
        when(rvs.score(1)).thenReturn(0.5f);

        final FlatVectorsScorer flatVectorsScorer = mock(FlatVectorsScorer.class);
        when(flatVectorsScorer.getRandomVectorScorer(eq(VectorSimilarityFunction.EUCLIDEAN), any(ByteVectorValues.class), eq(target)))
            .thenReturn(rvs);

        final FaissScorableByteVectorValues wrapper = new FaissScorableByteVectorValues(
            new PreDefinedByteVectorValues(VECTORS),
            flatVectorsScorer,
            VectorSimilarityFunction.EUCLIDEAN,
            null
        );

        // Advance the wrapper's own iterator
        final KnnVectorValues.DocIndexIterator wrapperIter = wrapper.iterator();
        wrapperIter.nextDoc();
        wrapperIter.nextDoc(); // now at doc 1

        // Scorer should start from the beginning (independent copy)
        final VectorScorer scorer = wrapper.scorer(target);
        final DocIdSetIterator scorerIter = scorer.iterator();
        assertEquals(0, scorerIter.nextDoc());
        assertEquals(1.0f, scorer.score(), 0.0f);

        assertEquals(1, scorerIter.nextDoc());
        assertEquals(0.5f, scorer.score(), 0.0f);
    }

    @SneakyThrows
    public void testScorerUsesOverrideIterator() {
        final byte[] target = new byte[] { 1, 2, 3, 4 };

        final RandomVectorScorer rvs = mock(RandomVectorScorer.class);
        when(rvs.score(0)).thenReturn(0.9f);

        final FlatVectorsScorer flatVectorsScorer = mock(FlatVectorsScorer.class);
        when(flatVectorsScorer.getRandomVectorScorer(eq(VectorSimilarityFunction.EUCLIDEAN), any(ByteVectorValues.class), eq(target)))
            .thenReturn(rvs);

        final KnnVectorValues.DocIndexIterator overrideIterator = mock(KnnVectorValues.DocIndexIterator.class);

        final FaissScorableByteVectorValues wrapper = new FaissScorableByteVectorValues(
            new PreDefinedByteVectorValues(VECTORS),
            flatVectorsScorer,
            VectorSimilarityFunction.EUCLIDEAN,
            overrideIterator
        );

        // iterator() should return the override iterator, not the delegate's
        assertSame(overrideIterator, wrapper.iterator());

        // scorer's iterator should be the same override instance
        final VectorScorer scorer = wrapper.scorer(target);
        assertSame(overrideIterator, scorer.iterator());
    }

    @SneakyThrows
    public void testGetSliceReturnsNullWhenDelegateDoesNotSupportIt() {
        final FaissScorableByteVectorValues wrapper = createWrapper();
        assertNull(wrapper.getSlice());
    }

    @SneakyThrows
    public void testGetSliceReturnsDelegateSlice() {
        final IndexInput expectedSlice = mock(IndexInput.class);
        final ByteVectorValues delegate = mock(ByteVectorValuesWithSlice.class);
        when(((HasIndexSlice) delegate).getSlice()).thenReturn(expectedSlice);

        final FaissScorableByteVectorValues wrapper = new FaissScorableByteVectorValues(
            delegate,
            mock(FlatVectorsScorer.class),
            VectorSimilarityFunction.EUCLIDEAN,
            null
        );

        assertSame(expectedSlice, wrapper.getSlice());
    }

    private static FaissScorableByteVectorValues createWrapper() {
        return new FaissScorableByteVectorValues(
            new PreDefinedByteVectorValues(VECTORS),
            mock(FlatVectorsScorer.class),
            VectorSimilarityFunction.EUCLIDEAN,
            null
        );
    }

    /**
     * Abstract class combining {@link ByteVectorValues} and {@link HasIndexSlice} for mocking.
     */
    private static abstract class ByteVectorValuesWithSlice extends ByteVectorValues implements HasIndexSlice {}
}
