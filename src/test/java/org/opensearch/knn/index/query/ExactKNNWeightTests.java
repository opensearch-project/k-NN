/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.SneakyThrows;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.query.iterators.KNNIterator;
import org.opensearch.knn.index.query.iterators.VectorIdsKNNIterator;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.apache.lucene.search.join.BitSetProducer;
import org.opensearch.knn.plugin.script.KNNScoringUtil;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.opensearch.knn.KNNRestTestCase.INDEX_NAME;

public class ExactKNNWeightTests extends KNNTestCase {

    private static final String FIELD_NAME = "test_field";
    private static final float[] QUERY_VECTOR = { 1.0f, 0.0f };
    private static final float BOOST = 2.0f;
    private static final String SPACE_TYPE = "innerproduct";

    @SneakyThrows
    public void testScorerSupplier_ReturnsLazyScorer() {
        ExactKNNFloatQuery mockQuery = mock(ExactKNNFloatQuery.class);
        when(mockQuery.getField()).thenReturn(FIELD_NAME);
        when(mockQuery.getQueryVector()).thenReturn(QUERY_VECTOR);
        when(mockQuery.getVectorDataType()).thenReturn(VectorDataType.FLOAT);

        ExactSearcher mockSearcher = mock(ExactSearcher.class);
        KNNIterator mockIterator = mock(KNNIterator.class);
        when(mockSearcher.createIterator(any(), any())).thenReturn(mockIterator);

        ExactKNNWeight.initialize(mockSearcher);

        ExactKNNWeight weight = new ExactKNNWeight(mockQuery, BOOST);
        LeafReaderContext mockContext = createMockLeafReaderContext();

        ScorerSupplier supplier = weight.scorerSupplier(mockContext);
        Scorer scorer = supplier.get(0);

        assertTrue(scorer instanceof KNNExactLazyScorer);
        verify(mockSearcher).createIterator(eq(mockContext), any());
    }

    @SneakyThrows
    public void testScorerSupplier_NullIterator_ReturnsEmptyScorer() {
        ExactKNNFloatQuery mockQuery = mock(ExactKNNFloatQuery.class);
        when(mockQuery.getVectorDataType()).thenReturn(VectorDataType.FLOAT);

        ExactSearcher mockSearcher = mock(ExactSearcher.class);
        when(mockSearcher.createIterator(any(), any())).thenReturn(null);

        ExactKNNWeight.initialize(mockSearcher);

        ExactKNNWeight weight = new ExactKNNWeight(mockQuery, BOOST);
        LeafReaderContext mockContext = createMockLeafReaderContext();

        ScorerSupplier supplier = weight.scorerSupplier(mockContext);
        Scorer scorer = supplier.get(0);

        assertEquals(NO_MORE_DOCS, scorer.iterator().nextDoc());
    }

    @SneakyThrows
    public void testLazyScorerIntegration() {
        ExactKNNFloatQuery mockQuery = mock(ExactKNNFloatQuery.class);
        when(mockQuery.getField()).thenReturn(FIELD_NAME);
        when(mockQuery.getQueryVector()).thenReturn(QUERY_VECTOR);
        when(mockQuery.getVectorDataType()).thenReturn(VectorDataType.FLOAT);

        KNNIterator mockIterator = mock(KNNIterator.class);
        when(mockIterator.nextDoc()).thenReturn(1, 5, NO_MORE_DOCS);
        when(mockIterator.score()).thenReturn(0.8f, 0.6f);

        ExactSearcher mockSearcher = mock(ExactSearcher.class);
        when(mockSearcher.createIterator(any(), any())).thenReturn(mockIterator);

        ExactKNNWeight.initialize(mockSearcher);

        ExactKNNWeight weight = new ExactKNNWeight(mockQuery, BOOST);
        LeafReaderContext mockContext = createMockLeafReaderContext();

        ScorerSupplier supplier = weight.scorerSupplier(mockContext);
        Scorer scorer = supplier.get(0);
        DocIdSetIterator iterator = scorer.iterator();

        assertEquals(1, iterator.nextDoc());
        assertEquals(1.6f, scorer.score(), 0.0f); // 0.8 * 2.0 boost

        assertEquals(5, iterator.nextDoc());
        assertEquals(1.2f, scorer.score(), 0.0f); // 0.6 * 2.0 boost

        assertEquals(NO_MORE_DOCS, iterator.nextDoc());
    }

    @SneakyThrows
    public void testLazyScoring_ScoreNotCalledUntilRequested() {
        ExactKNNFloatQuery mockQuery = mock(ExactKNNFloatQuery.class);
        when(mockQuery.getField()).thenReturn(FIELD_NAME);
        when(mockQuery.getQueryVector()).thenReturn(QUERY_VECTOR);
        when(mockQuery.getVectorDataType()).thenReturn(VectorDataType.FLOAT);

        KNNIterator mockIterator = mock(KNNIterator.class);
        when(mockIterator.nextDoc()).thenReturn(1, 2, NO_MORE_DOCS);
        when(mockIterator.score()).thenReturn(0.8f, 0.6f);

        ExactSearcher mockSearcher = mock(ExactSearcher.class);
        when(mockSearcher.createIterator(any(), any())).thenReturn(mockIterator);

        ExactKNNWeight.initialize(mockSearcher);

        ExactKNNWeight weight = new ExactKNNWeight(mockQuery, BOOST);
        LeafReaderContext mockContext = createMockLeafReaderContext();

        ScorerSupplier supplier = weight.scorerSupplier(mockContext);
        Scorer scorer = supplier.get(0);
        DocIdSetIterator iterator = scorer.iterator();

        // verifying score() was not called yet
        assertEquals(1, iterator.nextDoc());
        verify(mockIterator, never()).score();

        // verifying score() was called exactly once
        assertEquals(1.6f, scorer.score(), 0.0f);
        verify(mockIterator, times(1)).score();

        // verifying score() still only called once
        assertEquals(2, iterator.nextDoc());
        verify(mockIterator, times(1)).score();

        // verifying score() was called second time
        assertEquals(1.2f, scorer.score(), 0.0f);
        verify(mockIterator, times(2)).score();
    }

    @SneakyThrows
    public void testExplain() {
        ExactKNNFloatQuery mockQuery = mock(ExactKNNFloatQuery.class);
        when(mockQuery.getField()).thenReturn(FIELD_NAME);
        when(mockQuery.getQueryVector()).thenReturn(QUERY_VECTOR);
        when(mockQuery.getVectorDataType()).thenReturn(VectorDataType.FLOAT);

        ExactKNNWeight weight = new ExactKNNWeight(mockQuery, BOOST);
        LeafReaderContext mockContext = mock(LeafReaderContext.class);

        Explanation explanation = weight.explain(mockContext, 1, 0.5f);

        assertEquals(1.0f, explanation.getValue().floatValue(), 0.0f); // 0.5 * 2.0 boost
        assertTrue(explanation.getDescription().contains("exact k-NN search"));
        assertTrue(explanation.getDescription().contains(FIELD_NAME));
        verify(mockQuery).setExplain(true);
    }

    public void testIsCacheable() {
        ExactKNNFloatQuery mockQuery = mock(ExactKNNFloatQuery.class);
        ExactKNNWeight weight = new ExactKNNWeight(mockQuery, BOOST);

        assertTrue(weight.isCacheable(mock(LeafReaderContext.class)));
    }

    @SneakyThrows
    public void testExactSearchContextCreation() {
        ExactKNNFloatQuery mockQuery = mock(ExactKNNFloatQuery.class);
        when(mockQuery.getField()).thenReturn(FIELD_NAME);
        when(mockQuery.getQueryVector()).thenReturn(QUERY_VECTOR);
        when(mockQuery.getSpaceType()).thenReturn(SPACE_TYPE);
        when(mockQuery.getParentFilter()).thenReturn(null);
        when(mockQuery.getVectorDataType()).thenReturn(VectorDataType.FLOAT);

        ExactSearcher mockSearcher = mock(ExactSearcher.class);
        KNNIterator mockIterator = mock(KNNIterator.class);
        when(mockSearcher.createIterator(any(), any())).thenReturn(mockIterator);

        ExactKNNWeight.initialize(mockSearcher);

        ExactKNNWeight weight = new ExactKNNWeight(mockQuery, BOOST);
        LeafReaderContext mockContext = createMockLeafReaderContext();

        ScorerSupplier supplier = weight.scorerSupplier(mockContext);
        supplier.get(0);

        verify(mockSearcher).createIterator(eq(mockContext), any(ExactSearcher.ExactSearcherContext.class));
    }

    @SneakyThrows
    public void testNestedQueryWithParentFilter() {
        BitSetProducer mockParentFilter = mock(BitSetProducer.class);

        ExactKNNFloatQuery mockQuery = mock(ExactKNNFloatQuery.class);
        when(mockQuery.getField()).thenReturn(FIELD_NAME);
        when(mockQuery.getQueryVector()).thenReturn(QUERY_VECTOR);
        when(mockQuery.getSpaceType()).thenReturn(SPACE_TYPE);
        when(mockQuery.getParentFilter()).thenReturn(mockParentFilter);
        when(mockQuery.getVectorDataType()).thenReturn(VectorDataType.FLOAT);

        ExactSearcher mockSearcher = mock(ExactSearcher.class);
        KNNIterator mockIterator = mock(KNNIterator.class);
        when(mockSearcher.createIterator(any(), any())).thenReturn(mockIterator);

        ExactKNNWeight.initialize(mockSearcher);

        ExactKNNWeight weight = new ExactKNNWeight(mockQuery, BOOST);
        LeafReaderContext mockContext = createMockLeafReaderContext();

        ScorerSupplier supplier = weight.scorerSupplier(mockContext);
        Scorer scorer = supplier.get(0);

        assertTrue(scorer instanceof KNNExactLazyScorer);
        verify(mockSearcher).createIterator(eq(mockContext), any(ExactSearcher.ExactSearcherContext.class));
    }

    @SneakyThrows
    public void testExactSearch_thenCorrectDocOrderWithCorrectScores() {
        float[][] docVectors = {
            { 0.5f, 0.0f },  // inner product = 0.5 (second)
            { 1.0f, 0.0f },  // inner product = 1.0 (first)
            { 0.0f, 1.0f }   // inner product = 0.0 (third)
        };

        KNNFloatVectorValues vectorValues = mock(KNNFloatVectorValues.class);
        when(vectorValues.nextDoc()).thenReturn(0, 1, 2, NO_MORE_DOCS);
        when(vectorValues.getVector()).thenReturn(docVectors[0]).thenReturn(docVectors[1]).thenReturn(docVectors[2]);

        VectorIdsKNNIterator realIterator = new VectorIdsKNNIterator(null, QUERY_VECTOR, vectorValues, SpaceType.getSpace(SPACE_TYPE));

        float[] expectedScores = new float[docVectors.length];
        for (int i = 0; i < docVectors.length; i++) {
            expectedScores[i] = KNNWeight.normalizeScore(-KNNScoringUtil.innerProduct(QUERY_VECTOR, docVectors[i]));
        }

        ExactSearcher mockSearcher = mock(ExactSearcher.class);
        when(mockSearcher.createIterator(any(), any())).thenReturn(realIterator);

        final ExactKNNFloatQuery query = new ExactKNNFloatQuery(
            FIELD_NAME,
            SPACE_TYPE,
            INDEX_NAME,
            VectorDataType.FLOAT,
            null,
            false,
            QUERY_VECTOR
        );

        ExactKNNWeight.initialize(mockSearcher);

        ExactKNNWeight exactKNNWeight = new ExactKNNWeight(query, 1.0f);
        LeafReaderContext mockContext = createMockLeafReaderContext();

        ScorerSupplier supplier = exactKNNWeight.scorerSupplier(mockContext);
        Scorer scorer = supplier.get(0);
        DocIdSetIterator iterator = scorer.iterator();

        // verify that documents returned in ID order with correct scores
        assertEquals(0, iterator.nextDoc());
        assertEquals(expectedScores[0], scorer.score(), 0.001f);

        assertEquals(1, iterator.nextDoc());
        assertEquals(expectedScores[1], scorer.score(), 0.001f);

        assertEquals(2, iterator.nextDoc());
        assertEquals(expectedScores[2], scorer.score(), 0.001f);

        assertEquals(NO_MORE_DOCS, iterator.nextDoc());
    }

    private LeafReaderContext createMockLeafReaderContext() {
        LeafReaderContext mockContext = mock(LeafReaderContext.class);
        LeafReader mockReader = mock(LeafReader.class);
        when(mockContext.reader()).thenReturn(mockReader);
        when(mockReader.maxDoc()).thenReturn(100);
        return mockContext;
    }
}
