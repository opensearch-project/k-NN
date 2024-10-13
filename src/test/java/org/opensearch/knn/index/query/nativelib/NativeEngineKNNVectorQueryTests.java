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
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.util.Bits;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.mockito.invocation.InvocationOnMock;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.query.KNNQuery;
import org.opensearch.knn.index.query.KNNWeight;
import org.opensearch.knn.index.query.ResultUtil;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.test.OpenSearchTestCase;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;

import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.times;
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

    @InjectMocks
    private NativeEngineKnnVectorQuery objectUnderTest;

    private static ScoreMode scoreMode = ScoreMode.TOP_SCORES;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        openMocks(this);

        when(leaf1.reader()).thenReturn(leafReader1);
        when(leaf2.reader()).thenReturn(leafReader2);

        when(searcher.getIndexReader()).thenReturn(reader);
        when(knnQuery.createWeight(searcher, scoreMode, 1)).thenReturn(knnWeight);

        when(searcher.getTaskExecutor()).thenReturn(taskExecutor);
        when(taskExecutor.invokeAll(any())).thenAnswer(invocationOnMock -> {
            List<Callable<Map<Integer, Float>>> callables = invocationOnMock.getArgument(0);
            List<Map<Integer, Float>> results = new ArrayList<>();
            for (Callable<Map<Integer, Float>> callable : callables) {
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

        when(knnWeight.searchLeaf(leaf1, 4)).thenReturn(new HashMap<>(Map.of(0, 1.2f, 1, 5.1f, 2, 2.2f)));
        when(knnWeight.searchLeaf(leaf2, 4)).thenReturn(new HashMap<>(Map.of(4, 3.4f, 3, 5.1f)));

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
        int[] expectedDocs = { 0, 3, 4 };
        float[] expectedScores = { 1.2f, 5.1f, 3.4f };
        int[] findSegments = { 0, 1, 3 };
        Query expected = new DocAndScoreQuery(4, expectedDocs, expectedScores, findSegments, 1);

        // When
        Weight actual = objectUnderTest.createWeight(searcher, scoreMode, 1);

        // Then
        assertEquals(expected, actual.getQuery());
    }

    @SneakyThrows
    public void testRescoreWhenShardLevelRescoringEnabled() {
        // Given
        List<LeafReaderContext> leaves = List.of(leaf1, leaf2);
        when(reader.leaves()).thenReturn(leaves);

        int k = 2;
        int firstPassK = 3;
        Map<Integer, Float> initialLeaf1Results = new HashMap<>(Map.of(0, 21f, 1, 19f, 2, 17f));
        Map<Integer, Float> initialLeaf2Results = new HashMap<>(Map.of(0, 20f, 1, 18f, 2, 16f));
        Map<Integer, Float> rescoredLeaf1Results = new HashMap<>(Map.of(0, 18f, 1, 20f));
        Map<Integer, Float> rescoredLeaf2Results = new HashMap<>(Map.of(0, 21f));

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
            mockedKnnSettings.when(() -> KNNSettings.isShardLevelRescoringEnabledForDiskBasedVector(any())).thenReturn(true);

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
        List<LeafReaderContext> leaves = List.of(leaf1);
        when(reader.leaves()).thenReturn(leaves);
        when(knnWeight.searchLeaf(leaf1, 4)).thenReturn(new HashMap<>(Map.of(0, 1.2f, 1, 5.1f, 2, 2.2f)));
        when(knnQuery.getK()).thenReturn(4);

        when(indexReaderContext.id()).thenReturn(1);
        int[] expectedDocs = { 0, 1, 2 };
        float[] expectedScores = { 1.2f, 5.1f, 2.2f };
        int[] findSegments = { 0, 3 };
        Query expected = new DocAndScoreQuery(4, expectedDocs, expectedScores, findSegments, 1);

        // When
        Weight actual = objectUnderTest.createWeight(searcher, scoreMode, 1);

        // Then
        assertEquals(expected, actual.getQuery());
    }

    @SneakyThrows
    public void testNoMatch() {
        // Given
        List<LeafReaderContext> leaves = List.of(leaf1);
        when(reader.leaves()).thenReturn(leaves);
        when(knnWeight.searchLeaf(leaf1, 4)).thenReturn(Collections.emptyMap());
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
        int firstPassK = 3;
        Map<Integer, Float> initialLeaf1Results = new HashMap<>(Map.of(0, 21f, 1, 19f, 2, 17f, 3, 15f));
        Map<Integer, Float> initialLeaf2Results = new HashMap<>(Map.of(0, 20f, 1, 18f, 2, 16f, 3, 14f));
        Map<Integer, Float> rescoredLeaf1Results = new HashMap<>(Map.of(0, 18f, 1, 20f));
        Map<Integer, Float> rescoredLeaf2Results = new HashMap<>(Map.of(0, 21f));
        TopDocs topDocs1 = ResultUtil.resultMapToTopDocs(Map.of(1, 20f), 0);
        TopDocs topDocs2 = ResultUtil.resultMapToTopDocs(Map.of(0, 21f), 4);
        Query expected = new DocAndScoreQuery(2, new int[] { 1, 4 }, new float[] { 20f, 21f }, new int[] { 0, 4, 2 }, 1);

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
            mockedKnnSettings.when(() -> KNNSettings.isShardLevelRescoringEnabledForDiskBasedVector(any())).thenReturn(true);

            mockedResultUtil.when(() -> ResultUtil.reduceToTopK(any(), anyInt())).thenAnswer(InvocationOnMock::callRealMethod);
            mockedResultUtil.when(() -> ResultUtil.resultMapToTopDocs(eq(rescoredLeaf1Results), anyInt())).thenAnswer(t -> topDocs1);
            mockedResultUtil.when(() -> ResultUtil.resultMapToTopDocs(eq(rescoredLeaf2Results), anyInt())).thenAnswer(t -> topDocs2);
            try (MockedStatic<NativeEngineKnnVectorQuery> mockedStaticNativeKnnVectorQuery = mockStatic(NativeEngineKnnVectorQuery.class)) {
                mockedStaticNativeKnnVectorQuery.when(() -> NativeEngineKnnVectorQuery.findSegmentStarts(any(), any()))
                    .thenReturn(new int[] { 0, 4, 2 });
                Weight actual = objectUnderTest.createWeight(searcher, scoreMode, 1);
                assertEquals(expected, actual.getQuery());
            }
        }
    }
}
