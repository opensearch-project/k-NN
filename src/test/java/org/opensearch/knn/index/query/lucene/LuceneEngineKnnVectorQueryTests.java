/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucene;

import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.index.LeafReaderContext;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.Spy;
import org.opensearch.knn.index.query.lucenelib.InternalKnnFloatVectorQuery;
import org.opensearch.knn.index.query.lucenelib.InternalKnnByteVectorQuery;
import org.opensearch.test.OpenSearchTestCase;

import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.mock;
import static org.mockito.MockitoAnnotations.openMocks;

public class LuceneEngineKnnVectorQueryTests extends OpenSearchTestCase {

    @Mock
    IndexSearcher indexSearcher;

    @Mock
    Query luceneQuery;

    @Mock
    Weight weight;

    @Mock
    QueryVisitor queryVisitor;

    @Spy
    @InjectMocks
    LuceneEngineKnnVectorQuery objectUnderTest;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        openMocks(this);
        when(luceneQuery.rewrite(any(IndexSearcher.class))).thenReturn(luceneQuery);
        when(luceneQuery.createWeight(any(IndexSearcher.class), any(ScoreMode.class), anyFloat())).thenReturn(weight);
    }

    public void testRewrite() {
        objectUnderTest.rewrite(indexSearcher);
        objectUnderTest.rewrite(indexSearcher);
        objectUnderTest.rewrite(indexSearcher);
        verifyNoInteractions(luceneQuery);
        verify(objectUnderTest, times(3)).rewrite(indexSearcher);
    }

    public void testCreateWeight() throws Exception {
        objectUnderTest.rewrite(indexSearcher);
        objectUnderTest.rewrite(indexSearcher);
        objectUnderTest.rewrite(indexSearcher);
        verifyNoInteractions(luceneQuery);
        Weight actualWeight = objectUnderTest.createWeight(indexSearcher, ScoreMode.TOP_DOCS, 1.0f);
        verify(luceneQuery, times(1)).rewrite(indexSearcher);
        verify(objectUnderTest, times(3)).rewrite(indexSearcher);
        assertEquals(weight, actualWeight);
    }

    public void testVisit() {
        objectUnderTest.visit(queryVisitor);
        verify(queryVisitor).visitLeaf(objectUnderTest);
    }

    public void testEquals() {
        LuceneEngineKnnVectorQuery mainQuery = new LuceneEngineKnnVectorQuery(luceneQuery);
        LuceneEngineKnnVectorQuery otherQuery = new LuceneEngineKnnVectorQuery(luceneQuery);
        assertEquals(mainQuery, otherQuery);
        assertEquals(mainQuery, mainQuery);
        assertNotEquals(mainQuery, null);
        assertNotEquals(mainQuery, new Object());
        LuceneEngineKnnVectorQuery otherQuery2 = new LuceneEngineKnnVectorQuery(null);
        assertNotEquals(mainQuery, otherQuery2);
    }

    public void testHashCode() {
        LuceneEngineKnnVectorQuery mainQuery = new LuceneEngineKnnVectorQuery(luceneQuery);
        assertEquals(mainQuery.hashCode(), luceneQuery.hashCode());
    }

    public void testToString() {
        LuceneEngineKnnVectorQuery mainQuery = new LuceneEngineKnnVectorQuery(luceneQuery);
        assertEquals(mainQuery.toString(), luceneQuery.toString());
    }

    public void testCreateWeightWithInternalFloatVectorQuery() throws Exception {
        float[] vector = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        InternalKnnFloatVectorQuery floatVectorQuery = mock(InternalKnnFloatVectorQuery.class);
        InternalKnnFloatVectorQuery.initialize(null);

        when(floatVectorQuery.getField()).thenReturn("test-field");
        when(floatVectorQuery.getTargetCopy()).thenReturn(vector);
        when(floatVectorQuery.getK()).thenReturn(2);
        when(floatVectorQuery.getExactSearchSpaceType()).thenReturn("l2");
        when(floatVectorQuery.rewrite(any(IndexSearcher.class))).thenReturn(floatVectorQuery);

        TopDocs mockTopDocs = new TopDocs(
            new TotalHits(2, TotalHits.Relation.EQUAL_TO),
            new ScoreDoc[] { new ScoreDoc(0, 0.9f), new ScoreDoc(1, 0.7f) }
        );
        when(floatVectorQuery.searchLeaf(any(LeafReaderContext.class), anyInt(), any())).thenReturn(mockTopDocs);
        LuceneEngineKnnVectorQuery testQuery = new LuceneEngineKnnVectorQuery(floatVectorQuery);

        Weight weight = testQuery.createWeight(indexSearcher, ScoreMode.TOP_DOCS, 1.0f);
        verify(floatVectorQuery).rewrite(any(IndexSearcher.class));
        assertNotNull(weight);
        assertNotSame(this.weight, weight);

        LeafReaderContext context = mock(LeafReaderContext.class);
        Scorer scorer = weight.scorer(context);

        assertNotNull("Scorer should not be null", scorer);
        verify(floatVectorQuery).searchLeaf(eq(context), eq(2), any());

        Explanation explanation = weight.explain(context, 0);
        assertNotNull("Explanation should not be null", explanation);
        assertTrue(
            explanation.getDescription().contains("The type of search executed was KNN exact search")
                && explanation.toString().contains("l2")
        );
    }

    public void testCreateWeightWithInternalByteVectorQuery() throws Exception {
        byte[] vector = new byte[] { 1, 2, 3, 4 };
        InternalKnnByteVectorQuery byteVectorQuery = mock(InternalKnnByteVectorQuery.class);
        InternalKnnByteVectorQuery.initialize(null);

        when(byteVectorQuery.getField()).thenReturn("test-field");
        when(byteVectorQuery.getTargetCopy()).thenReturn(vector);
        when(byteVectorQuery.getK()).thenReturn(2);
        when(byteVectorQuery.getExactSearchSpaceType()).thenReturn("l2");
        when(byteVectorQuery.rewrite(any(IndexSearcher.class))).thenReturn(byteVectorQuery);

        TopDocs mockTopDocs = new TopDocs(
            new TotalHits(2, TotalHits.Relation.EQUAL_TO),
            new ScoreDoc[] { new ScoreDoc(0, 0.9f), new ScoreDoc(1, 0.7f) }
        );
        when(byteVectorQuery.searchLeaf(any(LeafReaderContext.class), anyInt(), any())).thenReturn(mockTopDocs);
        LuceneEngineKnnVectorQuery testQuery = new LuceneEngineKnnVectorQuery(byteVectorQuery);

        Weight weight = testQuery.createWeight(indexSearcher, ScoreMode.TOP_DOCS, 1.0f);
        verify(byteVectorQuery).rewrite(any(IndexSearcher.class));
        assertNotNull(weight);
        assertNotSame(this.weight, weight);

        LeafReaderContext context = mock(LeafReaderContext.class);
        Scorer scorer = weight.scorer(context);

        assertNotNull("Scorer should not be null", scorer);
        verify(byteVectorQuery).searchLeaf(eq(context), eq(2), any());

        Explanation explanation = weight.explain(context, 0);
        assertNotNull("Explanation should not be null", explanation);
        assertTrue(
            explanation.getDescription().contains("The type of search executed was KNN exact search")
                && explanation.toString().contains("l2")
        );
    }
}
