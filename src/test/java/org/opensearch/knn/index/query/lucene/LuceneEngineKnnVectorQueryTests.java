/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucene;

import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.CompositeReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.Weight;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.opensearch.knn.index.query.common.QueryUtils;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.Collections;

import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
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

    LuceneEngineKnnVectorQuery objectUnderTest;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        openMocks(this);
        objectUnderTest = new LuceneEngineKnnVectorQuery(luceneQuery, 10, 5);
        when(luceneQuery.rewrite(any(IndexSearcher.class))).thenReturn(luceneQuery);
        when(luceneQuery.createWeight(any(IndexSearcher.class), any(ScoreMode.class), anyFloat())).thenReturn(weight);
        when(luceneQuery.toString()).thenReturn("mocked query string");
    }

    public void testRewrite() {
        Query result1 = objectUnderTest.rewrite(indexSearcher);
        Query result2 = objectUnderTest.rewrite(indexSearcher);
        Query result3 = objectUnderTest.rewrite(indexSearcher);
        assertEquals(objectUnderTest, result1);
        assertEquals(objectUnderTest, result2);
        assertEquals(objectUnderTest, result3);
        verifyNoInteractions(luceneQuery);
    }

    public void testVisit() {
        objectUnderTest.visit(queryVisitor);
        verify(queryVisitor).visitLeaf(objectUnderTest);
    }

    public void testEquals() {
        LuceneEngineKnnVectorQuery mainQuery = new LuceneEngineKnnVectorQuery(luceneQuery, 10, 5);
        LuceneEngineKnnVectorQuery otherQuery = new LuceneEngineKnnVectorQuery(luceneQuery, 10, 5);
        assertEquals(mainQuery, otherQuery);
        assertEquals(mainQuery, mainQuery);
        assertNotEquals(mainQuery, null);
        assertNotEquals(mainQuery, new Object());
        LuceneEngineKnnVectorQuery otherQuery2 = new LuceneEngineKnnVectorQuery(null, 10, 5);
        assertNotEquals(mainQuery, otherQuery2);
        LuceneEngineKnnVectorQuery otherQuery3 = new LuceneEngineKnnVectorQuery(luceneQuery, 15, 5);
        assertNotEquals(mainQuery, otherQuery3);
        LuceneEngineKnnVectorQuery otherQuery4 = new LuceneEngineKnnVectorQuery(luceneQuery, 10, 8);
        assertNotEquals(mainQuery, otherQuery4);
    }

    public void testHashCode() {
        LuceneEngineKnnVectorQuery mainQuery1 = new LuceneEngineKnnVectorQuery(luceneQuery, 10, 5);
        LuceneEngineKnnVectorQuery mainQuery2 = new LuceneEngineKnnVectorQuery(luceneQuery, 10, 5);
        LuceneEngineKnnVectorQuery differentQuery = new LuceneEngineKnnVectorQuery(luceneQuery, 15, 5);

        assertEquals(mainQuery1.hashCode(), mainQuery2.hashCode());
        assertNotEquals(mainQuery1.hashCode(), differentQuery.hashCode());
    }

    public void testToString() {
        LuceneEngineKnnVectorQuery mainQuery = new LuceneEngineKnnVectorQuery(luceneQuery, 10, 5);
        String expected = "LuceneEngineKnnVectorQuery[luceneK=10, k=5, query=mocked query string]";
        assertEquals(expected, mainQuery.toString());
    }

    public void testCreateWeightWithoutReducing() throws IOException {
        // Test luceneK == k, should return original query
        LuceneEngineKnnVectorQuery queryNoReduce = new LuceneEngineKnnVectorQuery(luceneQuery, 5, 5);
        Weight actualWeight = queryNoReduce.createWeight(indexSearcher, ScoreMode.TOP_DOCS, 1.0f);
        verify(luceneQuery, times(1)).rewrite(indexSearcher);
        verify(luceneQuery, times(1)).createWeight(indexSearcher, ScoreMode.TOP_DOCS, 1.0f);
        assertEquals(weight, actualWeight);
    }

    public void testCreateWeightWithReduceToTopK() throws Exception {
        // Test luceneK > k, should reduce results to k
        DirectoryReader mockReader = mock(DirectoryReader.class);
        CompositeReaderContext mockContext = mock(CompositeReaderContext.class);
        TopDocs mockTopDocs = new TopDocs(
            new TotalHits(3, TotalHits.Relation.EQUAL_TO),
            new ScoreDoc[] { new ScoreDoc(0, 1.0f), new ScoreDoc(1, 0.8f), new ScoreDoc(2, 0.6f) }
        );

        when(indexSearcher.search(any(Query.class), eq(3))).thenReturn(mockTopDocs);
        when(indexSearcher.getIndexReader()).thenReturn(mockReader);
        when(mockReader.leaves()).thenReturn(Collections.emptyList());
        when(mockReader.getContext()).thenReturn(mockContext);

        try (MockedStatic<QueryUtils> queryUtilsMock = mockStatic(QueryUtils.class)) {
            QueryUtils mockQueryUtils = mock(QueryUtils.class);
            Query mockDocAndScoreQuery = mock(Query.class);
            Weight mockDocAndScoreWeight = mock(Weight.class);

            queryUtilsMock.when(QueryUtils::getInstance).thenReturn(mockQueryUtils);
            when(mockQueryUtils.createDocAndScoreQuery(any(), any())).thenReturn(mockDocAndScoreQuery);
            when(mockDocAndScoreQuery.createWeight(any(), any(), anyFloat())).thenReturn(mockDocAndScoreWeight);

            LuceneEngineKnnVectorQuery queryWithReduce = new LuceneEngineKnnVectorQuery(luceneQuery, 10, 3);
            Weight reducedWeight = queryWithReduce.createWeight(indexSearcher, ScoreMode.TOP_DOCS, 1.0f);

            verify(indexSearcher, times(1)).search(any(Query.class), eq(3));

            ArgumentCaptor<TopDocs> topDocsCaptor = ArgumentCaptor.forClass(TopDocs.class);
            verify(mockQueryUtils).createDocAndScoreQuery(eq(mockReader), topDocsCaptor.capture());

            TopDocs capturedTopDocs = topDocsCaptor.getValue();
            assertEquals(3, capturedTopDocs.scoreDocs.length);
            assertEquals(3, capturedTopDocs.totalHits.value());
            assertNotNull(reducedWeight);
        }
    }
}
