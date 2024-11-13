/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucenelib;

import junit.framework.TestCase;
import lombok.SneakyThrows;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.util.Bits;
import org.junit.Before;
import org.mockito.ArgumentCaptor;
import org.opensearch.knn.index.query.ResultUtil;
import org.opensearch.knn.index.query.common.QueryUtils;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class ExpandNestedEDocsQueryTests extends TestCase {
    private Executor executor;
    private TaskExecutor taskExecutor;

    @Before
    public void setUp() throws Exception {
        executor = Executors.newSingleThreadExecutor();
        taskExecutor = new TaskExecutor(executor);
    }

    @SneakyThrows
    public void testCreateWeight_whenCalled_thenSucceed() {
        LeafReaderContext leafReaderContext1 = mock(LeafReaderContext.class);
        LeafReaderContext leafReaderContext2 = mock(LeafReaderContext.class);
        List<LeafReaderContext> leafReaderContexts = Arrays.asList(leafReaderContext1, leafReaderContext2);

        IndexReader indexReader = mock(IndexReader.class);
        when(indexReader.leaves()).thenReturn(leafReaderContexts);

        Weight filterWeight = mock(Weight.class);

        IndexSearcher indexSearcher = mock(IndexSearcher.class);
        when(indexSearcher.getIndexReader()).thenReturn(indexReader);
        when(indexSearcher.getTaskExecutor()).thenReturn(taskExecutor);
        when(indexSearcher.createWeight(any(), eq(ScoreMode.COMPLETE_NO_SCORES), eq(1.0F))).thenReturn(filterWeight);

        Weight queryWeight = mock(Weight.class);
        ScoreMode scoreMode = mock(ScoreMode.class);
        float boost = 1.f;
        Query docAndScoreQuery = mock(Query.class);
        when(docAndScoreQuery.createWeight(indexSearcher, scoreMode, boost)).thenReturn(queryWeight);

        TopDocs topDocs1 = ResultUtil.resultMapToTopDocs(Map.of(1, 20f), 0);
        TopDocs topDocs2 = ResultUtil.resultMapToTopDocs(Map.of(0, 21f), 4);

        Query filterQuery = mock(Query.class);
        BitSetProducer parentFilter = mock(BitSetProducer.class);

        InternalNestedKnnVectorQuery internalQuery = mock(InternalNestedKnnVectorQuery.class);
        when(internalQuery.knnRewrite(indexSearcher)).thenReturn(docAndScoreQuery);
        when(internalQuery.getK()).thenReturn(2);
        when(internalQuery.knnExactSearch(any(), any())).thenReturn(topDocs1, topDocs2);
        when(internalQuery.getFilter()).thenReturn(filterQuery);
        when(internalQuery.getField()).thenReturn("field");
        when(internalQuery.getParentFilter()).thenReturn(parentFilter);

        Map<Integer, Float> initialLeaf1Results = new HashMap<>(Map.of(0, 19f, 1, 20f, 2, 17f, 3, 15f));
        Map<Integer, Float> initialLeaf2Results = new HashMap<>(Map.of(0, 21f, 1, 18f, 2, 16f, 3, 14f));
        List<Map<Integer, Float>> perLeafResults = Arrays.asList(initialLeaf1Results, initialLeaf2Results);

        Bits queryFilterBits = mock(Bits.class);
        DocIdSetIterator allSiblings = mock(DocIdSetIterator.class);
        when(allSiblings.nextDoc()).thenReturn(1, 2, DocIdSetIterator.NO_MORE_DOCS);

        Weight expectedWeight = mock(Weight.class);
        TopDocs topK = TopDocs.merge(2, new TopDocs[] { topDocs1, topDocs2 });
        Query finalQuery = mock(Query.class);
        when(finalQuery.createWeight(indexSearcher, scoreMode, boost)).thenReturn(expectedWeight);

        QueryUtils queryUtils = mock(QueryUtils.class);
        when(queryUtils.doSearch(indexSearcher, leafReaderContexts, queryWeight)).thenReturn(perLeafResults);
        when(queryUtils.createBits(any(), any())).thenReturn(queryFilterBits);
        when(queryUtils.getAllSiblings(any(), any(), any(), any())).thenReturn(allSiblings);
        when(queryUtils.createDocAndScoreQuery(eq(indexReader), any())).thenReturn(finalQuery);

        // Run
        ExpandNestedDocsQuery query = new ExpandNestedDocsQuery(internalQuery, queryUtils);
        Weight finalWeigh = query.createWeight(indexSearcher, scoreMode, 1.f);

        // Verify
        assertEquals(expectedWeight, finalWeigh);
        verify(queryUtils).createBits(leafReaderContext1, filterWeight);
        verify(queryUtils).createBits(leafReaderContext2, filterWeight);
        verify(queryUtils).getAllSiblings(leafReaderContext1, perLeafResults.get(0).keySet(), parentFilter, queryFilterBits);
        verify(queryUtils).getAllSiblings(leafReaderContext2, perLeafResults.get(1).keySet(), parentFilter, queryFilterBits);
        ArgumentCaptor<TopDocs> topDocsCaptor = ArgumentCaptor.forClass(TopDocs.class);
        verify(queryUtils).createDocAndScoreQuery(eq(indexReader), topDocsCaptor.capture());
        TopDocs capturedTopDocs = topDocsCaptor.getValue();
        assertEquals(topK.totalHits, capturedTopDocs.totalHits);
        for (int i = 0; i < topK.scoreDocs.length; i++) {
            assertEquals(topK.scoreDocs[i].doc, capturedTopDocs.scoreDocs[i].doc);
            assertEquals(topK.scoreDocs[i].score, capturedTopDocs.scoreDocs[i].score, 0.01f);
            assertEquals(topK.scoreDocs[i].shardIndex, capturedTopDocs.scoreDocs[i].shardIndex);
        }

        // Verify acceptedDocIds is intersection of allSiblings and filteredDocIds
        ArgumentCaptor<DocIdSetIterator> iteratorCaptor = ArgumentCaptor.forClass(DocIdSetIterator.class);
        verify(internalQuery, times(perLeafResults.size())).knnExactSearch(any(), iteratorCaptor.capture());
        assertEquals(1, iteratorCaptor.getValue().nextDoc());
        assertEquals(2, iteratorCaptor.getValue().nextDoc());
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iteratorCaptor.getValue().nextDoc());
    }
}
