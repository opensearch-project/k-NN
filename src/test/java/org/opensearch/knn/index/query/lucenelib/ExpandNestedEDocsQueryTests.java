/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucenelib;

import junit.framework.TestCase;
import lombok.SneakyThrows;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.FloatPoint;
import org.apache.lucene.index.*;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.Bits;
import org.junit.Before;
import org.mockito.ArgumentCaptor;
import org.opensearch.knn.index.query.ResultUtil;
import org.opensearch.knn.index.query.common.QueryUtils;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
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
        Directory directory = new ByteBuffersDirectory();
        IndexWriterConfig config = new IndexWriterConfig();
        try (IndexWriter writer = new IndexWriter(directory, config)) {
            // Add documents to simulate multiple segments
            Document doc1 = new Document();
            doc1.add(new FloatPoint("vector", 1.0f, 2.0f, 3.0f));
            writer.addDocument(doc1);
            Document doc2 = new Document();
            doc2.add(new FloatPoint("vector", 4.0f, 5.0f, 6.0f));
            writer.addDocument(doc2);
            // Force the creation of a second segment
            writer.flush();
            Document doc3 = new Document();
            doc3.add(new FloatPoint("vector", 7.0f, 8.0f, 9.0f));
            writer.addDocument(doc3);
            Document doc4 = new Document();
            doc4.add(new FloatPoint("vector", 10.0f, 11.0f, 12.0f));
            writer.addDocument(doc4);
            writer.commit();
        }

        IndexReader reader = DirectoryReader.open(directory);

        List<LeafReaderContext> leaves = reader.leaves();
        assertEquals(2, leaves.size()); // Ensure we have two segments
        LeafReaderContext leaf1 = leaves.get(0);
        LeafReaderContext leaf2 = leaves.get(1);

        Weight filterWeight = mock(Weight.class);

        IndexSearcher indexSearcher = mock(IndexSearcher.class);
        when(indexSearcher.getIndexReader()).thenReturn(reader);
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
        when(queryUtils.doSearch(indexSearcher, reader.leaves(), queryWeight)).thenReturn(perLeafResults);
        when(queryUtils.createBits(any(), any())).thenReturn(queryFilterBits);
        when(queryUtils.getAllSiblings(any(), any(), any(), any())).thenReturn(allSiblings);
        when(queryUtils.createDocAndScoreQuery(eq(reader), any())).thenReturn(finalQuery);

        // Run
        ExpandNestedDocsQuery query = ExpandNestedDocsQuery.builder()
            .internalNestedKnnVectorQuery(internalQuery)
            .queryUtils(queryUtils)
            .build();
        Weight finalWeigh = query.createWeight(indexSearcher, scoreMode, 1.f);

        // Verify
        assertEquals(expectedWeight, finalWeigh);
        verify(queryUtils).createBits(leaf1, filterWeight);
        verify(queryUtils).createBits(leaf2, filterWeight);
        verify(queryUtils).getAllSiblings(leaf1, perLeafResults.get(0).keySet(), parentFilter, queryFilterBits);
        verify(queryUtils).getAllSiblings(leaf2, perLeafResults.get(1).keySet(), parentFilter, queryFilterBits);
        ArgumentCaptor<TopDocs> topDocsCaptor = ArgumentCaptor.forClass(TopDocs.class);
        verify(queryUtils).createDocAndScoreQuery(eq(reader), topDocsCaptor.capture());
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

        // Without a rescore budget, the rescore step must not run.
        verify(internalQuery, times(0)).knnRescoreSearch(any(), any());
    }

    /**
     * When rescoring is enabled, ExpandNestedDocsQuery must first run a diversified full-precision rescore
     * (knnRescoreSearch) over the oversampled parent candidates, reduce to the top k parents, and only then
     * expand all of their child documents (knnExactSearch). This verifies the ordering and that the expansion
     * operates on the surviving parents rather than the full oversampled set.
     */
    @SneakyThrows
    public void testCreateWeight_whenRescoreEnabled_thenRescoreThenExpand() {
        Directory directory = new ByteBuffersDirectory();
        IndexWriterConfig config = new IndexWriterConfig();
        try (IndexWriter writer = new IndexWriter(directory, config)) {
            Document doc1 = new Document();
            doc1.add(new FloatPoint("vector", 1.0f, 2.0f, 3.0f));
            writer.addDocument(doc1);
            Document doc2 = new Document();
            doc2.add(new FloatPoint("vector", 4.0f, 5.0f, 6.0f));
            writer.addDocument(doc2);
            writer.flush();
            Document doc3 = new Document();
            doc3.add(new FloatPoint("vector", 7.0f, 8.0f, 9.0f));
            writer.addDocument(doc3);
            Document doc4 = new Document();
            doc4.add(new FloatPoint("vector", 10.0f, 11.0f, 12.0f));
            writer.addDocument(doc4);
            writer.commit();
        }

        IndexReader reader = DirectoryReader.open(directory);
        List<LeafReaderContext> leaves = reader.leaves();
        assertEquals(2, leaves.size());
        LeafReaderContext leaf1 = leaves.get(0);
        LeafReaderContext leaf2 = leaves.get(1);

        Weight filterWeight = mock(Weight.class);
        IndexSearcher indexSearcher = mock(IndexSearcher.class);
        when(indexSearcher.getIndexReader()).thenReturn(reader);
        when(indexSearcher.getTaskExecutor()).thenReturn(taskExecutor);
        when(indexSearcher.createWeight(any(), eq(ScoreMode.COMPLETE_NO_SCORES), eq(1.0F))).thenReturn(filterWeight);

        Weight queryWeight = mock(Weight.class);
        ScoreMode scoreMode = mock(ScoreMode.class);
        float boost = 1.f;
        Query docAndScoreQuery = mock(Query.class);
        when(docAndScoreQuery.createWeight(indexSearcher, scoreMode, boost)).thenReturn(queryWeight);

        Query filterQuery = mock(Query.class);
        BitSetProducer parentFilter = mock(BitSetProducer.class);

        // Rescore returns the best child per parent per leaf (segment-local doc ids). k = 2, so after rebasing
        // and merging across leaves, both parents survive (leaf1 doc 1 and leaf2 doc 0 -> global doc 4).
        TopDocs rescoreLeaf1 = ResultUtil.resultMapToTopDocs(Map.of(1, 20f), 0);
        TopDocs rescoreLeaf2 = ResultUtil.resultMapToTopDocs(Map.of(0, 21f), 0);
        // Expansion returns all children of the surviving parents as segment-local ids (retrieveAll rebases
        // them by docBase). Values are illustrative.
        TopDocs expandLeaf1 = ResultUtil.resultMapToTopDocs(Map.of(0, 20f, 1, 19f), 0);
        TopDocs expandLeaf2 = ResultUtil.resultMapToTopDocs(Map.of(0, 21f, 1, 18f), 0);

        InternalNestedKnnVectorQuery internalQuery = mock(InternalNestedKnnVectorQuery.class);
        when(internalQuery.knnRewrite(indexSearcher)).thenReturn(docAndScoreQuery);
        when(internalQuery.getK()).thenReturn(2);
        when(internalQuery.knnRescoreSearch(any(), any())).thenReturn(rescoreLeaf1, rescoreLeaf2);
        when(internalQuery.knnExactSearch(any(), any())).thenReturn(expandLeaf1, expandLeaf2);
        when(internalQuery.getFilter()).thenReturn(filterQuery);
        when(internalQuery.getField()).thenReturn("field");
        when(internalQuery.getParentFilter()).thenReturn(parentFilter);

        Map<Integer, Float> initialLeaf1Results = new HashMap<>(Map.of(0, 19f, 1, 20f, 2, 17f, 3, 15f));
        Map<Integer, Float> initialLeaf2Results = new HashMap<>(Map.of(0, 21f, 1, 18f, 2, 16f, 3, 14f));
        List<Map<Integer, Float>> perLeafResults = Arrays.asList(initialLeaf1Results, initialLeaf2Results);

        Bits queryFilterBits = mock(Bits.class);
        DocIdSetIterator allSiblings = mock(DocIdSetIterator.class);

        Weight expectedWeight = mock(Weight.class);
        Query finalQuery = mock(Query.class);
        when(finalQuery.createWeight(indexSearcher, scoreMode, boost)).thenReturn(expectedWeight);

        QueryUtils queryUtils = mock(QueryUtils.class);
        when(queryUtils.doSearch(indexSearcher, reader.leaves(), queryWeight)).thenReturn(perLeafResults);
        when(queryUtils.createBits(any(), any())).thenReturn(queryFilterBits);
        when(queryUtils.getAllSiblings(any(), any(), any(), any())).thenReturn(allSiblings);
        when(queryUtils.createDocAndScoreQuery(eq(reader), any())).thenReturn(finalQuery);

        // Run with a rescore budget
        ExpandNestedDocsQuery query = ExpandNestedDocsQuery.builder()
            .internalNestedKnnVectorQuery(internalQuery)
            .queryUtils(queryUtils)
            .rescoreK(4)
            .build();
        Weight finalWeigh = query.createWeight(indexSearcher, scoreMode, 1.f);

        assertEquals(expectedWeight, finalWeigh);

        // Rescore runs once per leaf over the oversampled candidates.
        verify(internalQuery, times(2)).knnRescoreSearch(any(), any());

        // Expansion runs once per surviving leaf, and it must expand the surviving parents (from rescore),
        // not the original oversampled candidate set. getAllSiblings is invoked twice per leaf: once during
        // rescore (over the oversampled candidates) and once during expansion (over the surviving parents).
        verify(internalQuery, times(perLeafResults.size())).knnExactSearch(any(), any());
        verify(queryUtils).getAllSiblings(eq(leaf1), eq(Set.of(1)), eq(parentFilter), any());
        verify(queryUtils).getAllSiblings(eq(leaf2), eq(Set.of(0)), eq(parentFilter), any());

        // All expanded children reach the final doc-and-score query (no truncation to k).
        ArgumentCaptor<TopDocs> topDocsCaptor = ArgumentCaptor.forClass(TopDocs.class);
        verify(queryUtils).createDocAndScoreQuery(eq(reader), topDocsCaptor.capture());
        assertEquals(4, topDocsCaptor.getValue().scoreDocs.length);
    }

    /**
     * equals/hashCode must account for rescoreK so that two otherwise-identical queries with different
     * rescore budgets are not treated as equal (and are not incorrectly served from a shared query cache).
     */
    @SneakyThrows
    public void testEqualsAndHashCode_whenRescoreKDiffers_thenNotEqual() {
        InternalNestedKnnVectorQuery internalQuery = mock(InternalNestedKnnVectorQuery.class);
        QueryUtils queryUtils = mock(QueryUtils.class);

        ExpandNestedDocsQuery noRescore = ExpandNestedDocsQuery.builder()
            .internalNestedKnnVectorQuery(internalQuery)
            .queryUtils(queryUtils)
            .build();
        ExpandNestedDocsQuery noRescoreCopy = ExpandNestedDocsQuery.builder()
            .internalNestedKnnVectorQuery(internalQuery)
            .queryUtils(queryUtils)
            .build();
        ExpandNestedDocsQuery withRescore = ExpandNestedDocsQuery.builder()
            .internalNestedKnnVectorQuery(internalQuery)
            .queryUtils(queryUtils)
            .rescoreK(4)
            .build();

        // Same internal query and default (no) rescore budget -> equal and same hashCode.
        assertEquals(noRescore, noRescoreCopy);
        assertEquals(noRescore.hashCode(), noRescoreCopy.hashCode());

        // Differing rescore budget -> not equal.
        assertFalse(noRescore.equals(withRescore));
        assertFalse(withRescore.equals(noRescore));

        // A query is equal to itself and not equal to an unrelated type.
        assertEquals(withRescore, withRescore);
        assertFalse(withRescore.equals("not a query"));
    }
}
