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
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.Spy;
import org.opensearch.test.OpenSearchTestCase;

import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
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
}
