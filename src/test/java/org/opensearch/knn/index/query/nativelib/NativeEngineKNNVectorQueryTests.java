/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.nativelib;

import lombok.SneakyThrows;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexReaderContext;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.util.Bits;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.mockito.invocation.InvocationOnMock;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.query.ExactSearcher;
import org.opensearch.knn.index.query.KNNQuery;
import org.opensearch.knn.index.query.KNNWeight;
import org.opensearch.knn.index.query.PerLeafResult;
import org.opensearch.knn.index.query.ResultUtil;
import org.opensearch.knn.index.query.common.QueryUtils;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.test.OpenSearchTestCase;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.mockito.MockitoAnnotations.openMocks;

public class NativeEngineKNNVectorQueryTests extends OpenSearchTestCase {

    @Mock
    private IndexSearcher searcher;
    @Mock
    private IndexReader reader;
    @Mock
    private KNNQuery knnQuery;
    @Mock
    private KNNWeight knnWeight;
    @Mock
    private TaskExecutor taskExecutor;
    @Mock
    private IndexReaderContext indexReaderContext;
    @Mock
    private LeafReaderContext leaf1;
    @Mock
    private LeafReaderContext leaf2;
    @Mock
    private LeafReader leafReader1;
    @Mock
    private LeafReader leafReader2;

    @Mock
    private ClusterService clusterService;

    private NativeEngineKnnVectorQuery objectUnderTest;

    private static ScoreMode scoreMode = ScoreMode.TOP_SCORES;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        openMocks(this);
        objectUnderTest = new NativeEngineKnnVectorQuery(knnQuery, QueryUtils.INSTANCE, false);
        when(leaf1.reader()).thenReturn(leafReader1);
        when(leaf2.reader()).thenReturn(leafReader2);

        when(searcher.getIndexReader()).thenReturn(reader);
        when(knnQuery.createWeight(searcher, scoreMode, 1)).thenReturn(knnWeight);

        when(searcher.getTaskExecutor()).thenReturn(taskExecutor);
        when(taskExecutor.invokeAll(any())).thenAnswer(invocationOnMock -> {
            List<Callable<PerLeafResult>> callables = invocationOnMock.getArgument(0);
            List<PerLeafResult> results = new ArrayList<>();
            for (Callable<PerLeafResult> callable : callables) {
                results.add(callable.call());
            }
            return results;
        });

        when(reader.getContext()).thenReturn(indexReaderContext);

        when(clusterService.state()).thenReturn(mock(ClusterState.class)); // Mock ClusterState

        // Set ClusterService in KNNSettings
        KNNSettings.state().setClusterService(clusterService);
        when(knnQuery.getQueryVector()).thenReturn(new float[] { 1.0f, 2.0f, 3.0f });  // Example vector

    }

    @SneakyThrows
    public void testMultiLeaf() {
        // Given
        List<LeafReaderContext> leaves = List.of(leaf1, leaf2);
        when(reader.leaves()).thenReturn(leaves);

        PerLeafResult leaf1Result = new PerLeafResult(null, new HashMap<>(Map.of(0, 1.2f, 1, 5.1f, 2, 2.2f)));
        PerLeafResult leaf2Result = new PerLeafResult(null, new HashMap<>(Map.of(4, 3.4f, 3, 5.1f)));

        when(knnWeight.searchLeaf(leaf1, 4)).thenReturn(leaf1Result);
        when(knnWeight.searchLeaf(leaf2, 4)).thenReturn(leaf2Result);

        // Making sure there is deleted docs in one of the segments
        Bits liveDocs = mock(Bits.class);
        when(leafReader1.getLiveDocs()).thenReturn(liveDocs);
        when(leafReader2.getLiveDocs()).thenReturn(null);

        when(liveDocs.get(anyInt())).thenReturn(true);
        when(liveDocs.get(2)).thenReturn(false);
        when(liveDocs.get(1)).thenReturn(false);

        // k=4 to make sure we get topk results even if docs are deleted/less in one of the leaves
        when(knnQuery.getK()).thenReturn(4);
        when(indexReaderContext.id()).thenReturn(1);

        Map<Integer, Float> leaf1ResultLive = Map.of(0, 1.2f);
        TopDocs[] topDocs = {
            ResultUtil.resultMapToTopDocs(leaf1ResultLive, leaf1.docBase),
            ResultUtil.resultMapToTopDocs(leaf2Result.getResult(), leaf2.docBase) };
        TopDocs expectedTopDocs = TopDocs.merge(4, topDocs);

        // When
        Weight actual = objectUnderTest.createWeight(searcher, scoreMode, 1);

        // Then
        Query expected = QueryUtils.INSTANCE.createDocAndScoreQuery(reader, expectedTopDocs);
        assertEquals(expected, actual.getQuery());
    }

    @SneakyThrows
    public void testRescoreWhenShardLevelRescoringEnabled() {
        // Given
        List<LeafReaderContext> leaves = List.of(leaf1, leaf2);
        when(reader.leaves()).thenReturn(leaves);

        int k = 2;
        PerLeafResult initialLeaf1Results = new PerLeafResult(null, new HashMap<>(Map.of(0, 21f, 1, 19f, 2, 17f)));
        PerLeafResult initialLeaf2Results = new PerLeafResult(null, new HashMap<>(Map.of(0, 20f, 1, 18f, 2, 16f)));
        Map<Integer, Float> rescoredLeaf1Results = new HashMap<>(Map.of(0, 18f, 1, 20f));
        Map<Integer, Float> rescoredLeaf2Results = new HashMap<>(Map.of(0, 21f));

        RescoreContext rescoreContext = RescoreContext.builder().oversampleFactor(1.5f).build();
        int firstPassK = rescoreContext.getFirstPassK(k, true, 1);
        when(knnQuery.getRescoreContext()).thenReturn(RescoreContext.builder().oversampleFactor(1.5f).build());
        when(knnQuery.getK()).thenReturn(k);
        when(knnWeight.getQuery()).thenReturn(knnQuery);
        when(knnWeight.searchLeaf(leaf1, firstPassK)).thenReturn(initialLeaf1Results);
        when(knnWeight.searchLeaf(leaf2, firstPassK)).thenReturn(initialLeaf2Results);
        when(knnWeight.exactSearch(eq(leaf1), any())).thenReturn(rescoredLeaf1Results);
        when(knnWeight.exactSearch(eq(leaf2), any())).thenReturn(rescoredLeaf2Results);

        try (
            MockedStatic<KNNSettings> mockedKnnSettings = mockStatic(KNNSettings.class);
            MockedStatic<ResultUtil> mockedResultUtil = mockStatic(ResultUtil.class)
        ) {

            // When shard-level re-scoring is enabled
            mockedKnnSettings.when(() -> KNNSettings.isShardLevelRescoringDisabledForDiskBasedVector(any())).thenReturn(false);

            // Mock ResultUtil to return valid TopDocs
            mockedResultUtil.when(() -> ResultUtil.resultMapToTopDocs(any(), anyInt()))
                .thenReturn(new TopDocs(new TotalHits(0, TotalHits.Relation.EQUAL_TO), new ScoreDoc[0]));
            mockedResultUtil.when(() -> ResultUtil.reduceToTopK(any(), anyInt())).thenCallRealMethod();

            // When
            Weight actual = objectUnderTest.createWeight(searcher, scoreMode, 1);

            // Then
            mockedResultUtil.verify(() -> ResultUtil.reduceToTopK(any(), anyInt()), times(2));
            assertNotNull(actual);
        }
    }

    @SneakyThrows
    public void testSingleLeaf() {
        // Given
        int k = 4;
        float boost = 1;
        PerLeafResult leaf1Result = new PerLeafResult(null, new HashMap<>(Map.of(0, 1.2f, 1, 5.1f, 2, 2.2f)));
        List<LeafReaderContext> leaves = List.of(leaf1);
        when(reader.leaves()).thenReturn(leaves);
        when(knnWeight.searchLeaf(leaf1, k)).thenReturn(leaf1Result);
        when(knnQuery.getK()).thenReturn(k);
        when(indexReaderContext.id()).thenReturn(1);
        TopDocs expectedTopDocs = ResultUtil.resultMapToTopDocs(leaf1Result.getResult(), leaf1.docBase);

        // When
        Weight actual = objectUnderTest.createWeight(searcher, scoreMode, boost);

        // Then
        Query expected = QueryUtils.INSTANCE.createDocAndScoreQuery(reader, expectedTopDocs);
        assertEquals(expected, actual.getQuery());
    }

    @SneakyThrows
    public void testNoMatch() {
        // Given
        List<LeafReaderContext> leaves = List.of(leaf1);
        when(reader.leaves()).thenReturn(leaves);
        when(knnWeight.searchLeaf(leaf1, 4)).thenReturn(PerLeafResult.EMPTY_RESULT);
        when(knnQuery.getK()).thenReturn(4);

        // When
        Weight actual = objectUnderTest.createWeight(searcher, scoreMode, 1);

        // Then
        assertEquals(new MatchNoDocsQuery(), actual.getQuery());
    }

    @SneakyThrows
    public void testRescore() {
        // Given
        List<LeafReaderContext> leaves = List.of(leaf1, leaf2);
        when(reader.leaves()).thenReturn(leaves);

        int k = 2;
        int firstPassK = 100;
        PerLeafResult initialLeaf1Results = new PerLeafResult(null, new HashMap<>(Map.of(0, 21f, 1, 19f, 2, 17f, 3, 15f)));
        PerLeafResult initialLeaf2Results = new PerLeafResult(null, new HashMap<>(Map.of(0, 20f, 1, 18f, 2, 16f, 3, 14f)));
        Map<Integer, Float> rescoredLeaf1Results = new HashMap<>(Map.of(0, 18f, 1, 20f));
        Map<Integer, Float> rescoredLeaf2Results = new HashMap<>(Map.of(0, 21f));
        TopDocs topDocs1 = ResultUtil.resultMapToTopDocs(Map.of(1, 20f), 0);
        TopDocs topDocs2 = ResultUtil.resultMapToTopDocs(Map.of(0, 21f), 4);
        when(indexReaderContext.id()).thenReturn(1);
        when(knnQuery.getRescoreContext()).thenReturn(RescoreContext.builder().oversampleFactor(1.5f).build());
        when(knnQuery.getK()).thenReturn(k);
        when(knnWeight.getQuery()).thenReturn(knnQuery);
        when(knnWeight.searchLeaf(leaf1, firstPassK)).thenReturn(initialLeaf1Results);
        when(knnWeight.searchLeaf(leaf2, firstPassK)).thenReturn(initialLeaf2Results);

        when(knnWeight.exactSearch(eq(leaf1), any())).thenReturn(rescoredLeaf1Results);
        when(knnWeight.exactSearch(eq(leaf2), any())).thenReturn(rescoredLeaf2Results);

        try (
            MockedStatic<KNNSettings> mockedKnnSettings = mockStatic(KNNSettings.class);
            MockedStatic<ResultUtil> mockedResultUtil = mockStatic(ResultUtil.class)
        ) {

            // When shard-level re-scoring is enabled
            mockedKnnSettings.when(() -> KNNSettings.isShardLevelRescoringDisabledForDiskBasedVector(any())).thenReturn(false);

            mockedResultUtil.when(() -> ResultUtil.reduceToTopK(any(), anyInt())).thenAnswer(InvocationOnMock::callRealMethod);
            mockedResultUtil.when(() -> ResultUtil.resultMapToDocIds(any(), anyInt())).thenAnswer(InvocationOnMock::callRealMethod);

            mockedResultUtil.when(() -> ResultUtil.resultMapToTopDocs(eq(rescoredLeaf1Results), anyInt())).thenAnswer(t -> topDocs1);
            mockedResultUtil.when(() -> ResultUtil.resultMapToTopDocs(eq(rescoredLeaf2Results), anyInt())).thenAnswer(t -> topDocs2);

            // Run
            Weight actual = objectUnderTest.createWeight(searcher, scoreMode, 1);

            // Verify
            TopDocs[] topDocs = { topDocs1, topDocs2 };
            TopDocs expectedTopDocs = TopDocs.merge(k, topDocs);
            Query expected = QueryUtils.INSTANCE.createDocAndScoreQuery(reader, expectedTopDocs);
            assertEquals(expected, actual.getQuery());
        }
    }

    @SneakyThrows
    public void testExpandNestedDocs() {
        List<LeafReaderContext> leafReaderContexts = Arrays.asList(leaf1, leaf2);
        when(reader.leaves()).thenReturn(leafReaderContexts);
        Bits queryFilterBits = mock(Bits.class);
        PerLeafResult initialLeaf1Results = new PerLeafResult(queryFilterBits, new HashMap<>(Map.of(0, 19f, 1, 20f, 2, 17f, 3, 15f)));
        PerLeafResult initialLeaf2Results = new PerLeafResult(queryFilterBits, new HashMap<>(Map.of(0, 21f, 1, 18f, 2, 16f, 3, 14f)));
        List<Map<Integer, Float>> perLeafResults = Arrays.asList(initialLeaf1Results.getResult(), initialLeaf2Results.getResult());

        Map<Integer, Float> exactSearchLeaf1Result = new HashMap<>(Map.of(1, 20f));
        Map<Integer, Float> exactSearchLeaf2Result = new HashMap<>(Map.of(0, 21f));

        TopDocs topDocs1 = ResultUtil.resultMapToTopDocs(exactSearchLeaf1Result, 0);
        TopDocs topDocs2 = ResultUtil.resultMapToTopDocs(exactSearchLeaf2Result, 0);
        TopDocs topK = TopDocs.merge(2, new TopDocs[] { topDocs1, topDocs2 });

        int k = 2;
        when(knnQuery.getRescoreContext()).thenReturn(null);
        when(knnQuery.getK()).thenReturn(k);

        BitSetProducer parentFilter = mock(BitSetProducer.class);
        when(knnQuery.getParentsFilter()).thenReturn(parentFilter);
        when(knnWeight.searchLeaf(leaf1, k)).thenReturn(initialLeaf1Results);
        when(knnWeight.searchLeaf(leaf2, k)).thenReturn(initialLeaf2Results);
        when(knnWeight.exactSearch(any(), any())).thenReturn(exactSearchLeaf1Result, exactSearchLeaf2Result);
        Weight filterWeight = mock(Weight.class);
        when(knnWeight.getFilterWeight()).thenReturn(filterWeight);

        DocIdSetIterator allSiblings = mock(DocIdSetIterator.class);
        when(allSiblings.nextDoc()).thenReturn(1, 2, DocIdSetIterator.NO_MORE_DOCS);

        Weight expectedWeight = mock(Weight.class);
        Query finalQuery = mock(Query.class);
        when(finalQuery.createWeight(searcher, scoreMode, 1)).thenReturn(expectedWeight);

        QueryUtils queryUtils = mock(QueryUtils.class);
        when(queryUtils.getAllSiblings(any(), any(), any(), any())).thenReturn(allSiblings);
        when(queryUtils.createDocAndScoreQuery(eq(reader), any())).thenReturn(finalQuery);

        // Run
        NativeEngineKnnVectorQuery query = new NativeEngineKnnVectorQuery(knnQuery, queryUtils, true);
        Weight finalWeigh = query.createWeight(searcher, scoreMode, 1.f);

        // Verify
        assertEquals(expectedWeight, finalWeigh);
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
        ArgumentCaptor<ExactSearcher.ExactSearcherContext> contextCaptor = ArgumentCaptor.forClass(
            ExactSearcher.ExactSearcherContext.class
        );
        verify(knnWeight, times(perLeafResults.size())).exactSearch(any(), contextCaptor.capture());
        assertEquals(1, contextCaptor.getValue().getMatchedDocsIterator().nextDoc());
        assertEquals(2, contextCaptor.getValue().getMatchedDocsIterator().nextDoc());
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, contextCaptor.getValue().getMatchedDocsIterator().nextDoc());
    }
}
