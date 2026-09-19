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
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.Bits;
import org.junit.Before;
import org.mockito.ArgumentCaptor;
import org.opensearch.knn.index.query.ResultUtil;
import org.opensearch.knn.index.query.common.QueryUtils;
import org.opensearch.knn.index.query.exactsearch.ExactSearcher;
import org.opensearch.knn.index.query.rescore.RescoreContext;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import static org.junit.Assert.assertNotEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.atLeast;
import static org.mockito.Mockito.never;
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

        ExactSearcher exactSearcher = mock(ExactSearcher.class);

        // Run
        ExpandNestedDocsQuery query = new ExpandNestedDocsQuery.ExpandNestedDocsQueryBuilder().internalNestedKnnVectorQuery(internalQuery)
            .queryUtils(queryUtils)
            .exactSearcher(exactSearcher)
            .build();
        Weight finalWeigh = query.createWeight(indexSearcher, scoreMode, 1.f);

        // Verify
        assertEquals(expectedWeight, finalWeigh);
        // No rescore context, so the full precision rescore pass must not run at all
        verify(exactSearcher, never()).searchLeaf(any(), any());
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
    }

    /**
     * Regression test for https://github.com/opensearch-project/k-NN/issues/3125.
     *
     * With rescoring enabled the query must re-score the oversampled candidates on full precision vectors,
     * collapse each parent group to its best child, and cut to k <b>parent</b> documents before expanding the
     * children. This asserts the ordering: the expansion only sees the parents that survived the cut, so the
     * final result cannot be truncated to k child documents.
     */
    @SneakyThrows
    public void testCreateWeight_whenRescoreEnabled_thenRescoreAndCutToTopKParentsBeforeExpanding() {
        Directory directory = new ByteBuffersDirectory();
        try (IndexWriter writer = new IndexWriter(directory, new IndexWriterConfig())) {
            Document doc1 = new Document();
            doc1.add(new FloatPoint("vector", 1.0f, 2.0f, 3.0f));
            writer.addDocument(doc1);
            writer.flush();
            Document doc2 = new Document();
            doc2.add(new FloatPoint("vector", 4.0f, 5.0f, 6.0f));
            writer.addDocument(doc2);
            writer.commit();
        }

        IndexReader reader = DirectoryReader.open(directory);
        List<LeafReaderContext> leaves = reader.leaves();
        assertEquals(2, leaves.size());
        LeafReaderContext leaf1 = leaves.get(0);
        LeafReaderContext leaf2 = leaves.get(1);

        int k = 2;
        int rescoreK = 4;
        float[] queryVector = new float[] { 1.0f, 2.0f, 3.0f };

        Weight filterWeight = mock(Weight.class);
        IndexSearcher indexSearcher = mock(IndexSearcher.class);
        when(indexSearcher.getIndexReader()).thenReturn(reader);
        when(indexSearcher.getTaskExecutor()).thenReturn(taskExecutor);
        when(indexSearcher.createWeight(any(), eq(ScoreMode.COMPLETE_NO_SCORES), eq(1.0F))).thenReturn(filterWeight);

        Weight queryWeight = mock(Weight.class);
        ScoreMode scoreMode = mock(ScoreMode.class);
        Query docAndScoreQuery = mock(Query.class);
        when(docAndScoreQuery.createWeight(indexSearcher, scoreMode, 1.f)).thenReturn(queryWeight);

        BitSetProducer parentFilter = mock(BitSetProducer.class);
        InternalNestedKnnVectorQuery internalQuery = mock(InternalNestedKnnVectorQuery.class);
        when(internalQuery.knnRewrite(indexSearcher)).thenReturn(docAndScoreQuery);
        when(internalQuery.getK()).thenReturn(k);
        when(internalQuery.getFilter()).thenReturn(mock(Query.class));
        when(internalQuery.getField()).thenReturn("field");
        when(internalQuery.getParentFilter()).thenReturn(parentFilter);

        // Oversampled candidate pool: two candidate parents per leaf.
        List<Map<Integer, Float>> perLeafResults = Arrays.asList(
            new HashMap<>(Map.of(1, 20f, 5, 19f)),
            new HashMap<>(Map.of(0, 21f, 3, 18f))
        );

        Bits queryFilterBits = mock(Bits.class);
        DocIdSetIterator allSiblings = mock(DocIdSetIterator.class);
        QueryUtils queryUtils = mock(QueryUtils.class);
        when(queryUtils.doSearch(indexSearcher, reader.leaves(), queryWeight)).thenReturn(perLeafResults);
        when(queryUtils.createBits(any(), any())).thenReturn(queryFilterBits);
        when(queryUtils.getAllSiblings(any(), any(), any(), any())).thenReturn(allSiblings);

        // Rescore collapses each leaf to one row per parent. Doc 5 in leaf1 and doc 3 in leaf2 lose on full
        // precision scores and must be dropped by the cut to k=2 parents.
        // Stubs are keyed on the leaf rather than on call order, because TaskExecutor#invokeAll does not
        // promise to run the tasks in the order they were submitted.
        TopDocs rescoredLeaf1 = topDocs(new ScoreDoc(1, 30f), new ScoreDoc(5, 5f));
        TopDocs rescoredLeaf2 = topDocs(new ScoreDoc(0, 10f), new ScoreDoc(3, 4f));
        // Expansion returns every child of the surviving parents; there must be no cut after this point.
        TopDocs expandedLeaf1 = topDocs(new ScoreDoc(0, 30f), new ScoreDoc(1, 28f), new ScoreDoc(2, 27f));
        TopDocs expandedLeaf2 = topDocs(new ScoreDoc(0, 10f), new ScoreDoc(1, 9f), new ScoreDoc(2, 8f));
        // Both stages go through the exact searcher when rescoring. The collapsing stage is the one that carries
        // a parent filter, so that is what tells the two apart.
        ExactSearcher exactSearcher = mock(ExactSearcher.class);
        when(exactSearcher.searchLeaf(any(), any())).thenAnswer(invocation -> {
            boolean isFirstLeaf = ((LeafReaderContext) invocation.getArgument(0)).ord == 0;
            ExactSearcher.ExactSearcherContext context = invocation.getArgument(1);
            if (context.getParentsFilter() != null) {
                return isFirstLeaf ? rescoredLeaf1 : rescoredLeaf2;
            }
            return isFirstLeaf ? expandedLeaf1 : expandedLeaf2;
        });

        Weight expectedWeight = mock(Weight.class);
        Query finalQuery = mock(Query.class);
        when(finalQuery.createWeight(indexSearcher, scoreMode, 1.f)).thenReturn(expectedWeight);
        when(queryUtils.createDocAndScoreQuery(eq(reader), any())).thenReturn(finalQuery);

        // Run
        ExpandNestedDocsQuery query = new ExpandNestedDocsQuery.ExpandNestedDocsQueryBuilder().internalNestedKnnVectorQuery(internalQuery)
            .queryUtils(queryUtils)
            .rescoreK(rescoreK)
            .floatQueryVector(queryVector)
            .exactSearcher(exactSearcher)
            .build();
        Weight finalWeight = query.createWeight(indexSearcher, scoreMode, 1.f);

        assertEquals(expectedWeight, finalWeight);

        // Two collapsing searches plus two expansion searches, one of each per leaf. Lucene's own exact search
        // must not be used at all here, because it would score the children on quantized vectors.
        ArgumentCaptor<ExactSearcher.ExactSearcherContext> contextCaptor = ArgumentCaptor.forClass(
            ExactSearcher.ExactSearcherContext.class
        );
        verify(exactSearcher, times(4)).searchLeaf(any(), contextCaptor.capture());
        verify(internalQuery, never()).knnExactSearch(any(), any());

        List<ExactSearcher.ExactSearcherContext> collapsing = contextCaptor.getAllValues()
            .stream()
            .filter(context -> context.getParentsFilter() != null)
            .toList();
        List<ExactSearcher.ExactSearcherContext> expanding = contextCaptor.getAllValues()
            .stream()
            .filter(context -> context.getParentsFilter() == null)
            .toList();
        assertEquals(2, collapsing.size());
        assertEquals(2, expanding.size());

        // Every search scores full precision vectors, and every one targets the right field and query vector
        for (ExactSearcher.ExactSearcherContext context : contextCaptor.getAllValues()) {
            assertFalse(context.isUseQuantizedVectorsForSearch());
            assertEquals("field", context.getField());
            assertTrue(Arrays.equals(queryVector, context.getFloatQueryVector()));
        }
        // The collapsing stage cuts to k parents; the expansion stage keeps every sibling it is given
        for (ExactSearcher.ExactSearcherContext context : collapsing) {
            assertEquals(parentFilter, context.getParentsFilter());
            assertEquals(k, context.getK());
        }

        // The rescore stage saw the whole oversampled pool ...
        verify(queryUtils).getAllSiblings(leaf1, Set.of(1, 5), parentFilter, queryFilterBits);
        verify(queryUtils).getAllSiblings(leaf2, Set.of(0, 3), parentFilter, queryFilterBits);
        // ... but the expansion only saw the parents that survived the cut to k.
        verify(queryUtils).getAllSiblings(leaf1, Set.of(1), parentFilter, queryFilterBits);
        verify(queryUtils).getAllSiblings(leaf2, Set.of(0), parentFilter, queryFilterBits);

        // All children of both surviving parents are returned; nothing is truncated to k.
        ArgumentCaptor<TopDocs> topDocsCaptor = ArgumentCaptor.forClass(TopDocs.class);
        verify(queryUtils).createDocAndScoreQuery(eq(reader), topDocsCaptor.capture());
        assertEquals(6, topDocsCaptor.getValue().scoreDocs.length);
    }

    @SneakyThrows
    public void testCreateWeight_whenRescoreEnabledAndNoCandidates_thenSkipRescoreForThatLeaf() {
        Directory directory = new ByteBuffersDirectory();
        try (IndexWriter writer = new IndexWriter(directory, new IndexWriterConfig())) {
            Document doc = new Document();
            doc.add(new FloatPoint("vector", 1.0f, 2.0f, 3.0f));
            writer.addDocument(doc);
            writer.commit();
        }

        IndexReader reader = DirectoryReader.open(directory);
        assertEquals(1, reader.leaves().size());

        IndexSearcher indexSearcher = mock(IndexSearcher.class);
        when(indexSearcher.getIndexReader()).thenReturn(reader);
        when(indexSearcher.getTaskExecutor()).thenReturn(taskExecutor);

        Weight queryWeight = mock(Weight.class);
        ScoreMode scoreMode = mock(ScoreMode.class);
        Query docAndScoreQuery = mock(Query.class);
        when(docAndScoreQuery.createWeight(indexSearcher, scoreMode, 1.f)).thenReturn(queryWeight);

        InternalNestedKnnVectorQuery internalQuery = mock(InternalNestedKnnVectorQuery.class);
        when(internalQuery.knnRewrite(indexSearcher)).thenReturn(docAndScoreQuery);
        when(internalQuery.getK()).thenReturn(2);
        // Null filter keeps getFilterWeight from touching the searcher
        when(internalQuery.getFilter()).thenReturn(null);
        when(internalQuery.getField()).thenReturn("field");
        when(internalQuery.getParentFilter()).thenReturn(mock(BitSetProducer.class));
        when(internalQuery.knnExactSearch(any(), any())).thenReturn(topDocs());

        QueryUtils queryUtils = mock(QueryUtils.class);
        when(queryUtils.doSearch(indexSearcher, reader.leaves(), queryWeight)).thenReturn(List.of(Map.<Integer, Float>of()));
        when(queryUtils.getAllSiblings(any(), any(), any(), any())).thenReturn(DocIdSetIterator.empty());

        ExactSearcher exactSearcher = mock(ExactSearcher.class);
        when(exactSearcher.searchLeaf(any(), any())).thenReturn(topDocs());

        ExpandNestedDocsQuery query = new ExpandNestedDocsQuery.ExpandNestedDocsQueryBuilder().internalNestedKnnVectorQuery(internalQuery)
            .queryUtils(queryUtils)
            .rescoreK(4)
            .floatQueryVector(new float[] { 1.0f, 2.0f, 3.0f })
            .exactSearcher(exactSearcher)
            .build();

        Weight weight = query.createWeight(indexSearcher, scoreMode, 1.f);

        // With no candidates there is nothing to collapse, so the rescore stage is skipped for that leaf
        ArgumentCaptor<ExactSearcher.ExactSearcherContext> contextCaptor = ArgumentCaptor.forClass(
            ExactSearcher.ExactSearcherContext.class
        );
        verify(exactSearcher, atLeast(0)).searchLeaf(any(), contextCaptor.capture());
        assertTrue(
            "no collapsing search should run for a leaf with no candidates",
            contextCaptor.getAllValues().stream().noneMatch(context -> context.getParentsFilter() != null)
        );
        assertNotNull(weight);
    }

    public void testEquals_whenRescoreKDiffers_thenNotEqual() {
        InternalNestedKnnVectorQuery internalQuery = mock(InternalNestedKnnVectorQuery.class);
        QueryUtils queryUtils = mock(QueryUtils.class);

        ExpandNestedDocsQuery noRescore = new ExpandNestedDocsQuery.ExpandNestedDocsQueryBuilder().internalNestedKnnVectorQuery(
            internalQuery
        ).queryUtils(queryUtils).rescoreK(RescoreContext.NO_RESCORE_NEEDED).build();
        ExpandNestedDocsQuery rescore = new ExpandNestedDocsQuery.ExpandNestedDocsQueryBuilder().internalNestedKnnVectorQuery(internalQuery)
            .queryUtils(queryUtils)
            .rescoreK(10)
            .build();
        ExpandNestedDocsQuery sameRescore = new ExpandNestedDocsQuery.ExpandNestedDocsQueryBuilder().internalNestedKnnVectorQuery(
            internalQuery
        ).queryUtils(queryUtils).rescoreK(10).build();

        assertNotEquals(noRescore, rescore);
        assertNotEquals(noRescore.hashCode(), rescore.hashCode());
        assertEquals(rescore, sameRescore);
        assertEquals(rescore.hashCode(), sameRescore.hashCode());
    }

    private static TopDocs topDocs(ScoreDoc... scoreDocs) {
        return new TopDocs(new TotalHits(scoreDocs.length, TotalHits.Relation.EQUAL_TO), scoreDocs);
    }
}
