/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.nativelib;

import lombok.SneakyThrows;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.FloatPoint;
import org.apache.lucene.index.*;
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
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.analysis.MockAnalyzer;
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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;
import static org.mockito.MockitoAnnotations.openMocks;
import static org.opensearch.knn.utils.TopDocsTestUtils.buildTopDocs;
import static org.opensearch.knn.utils.TopDocsTestUtils.convertTopDocsToMap;

public class NativeEngineKNNVectorQueryTests extends OpenSearchTestCase {

    @Mock
    private IndexSearcher searcher;
    private IndexReader reader;
    private Directory directory;
    private DirectoryReader directoryReader;
    @Mock
    private KNNQuery knnQuery;
    @Mock
    private KNNWeight knnWeight;
    @Mock
    private TaskExecutor taskExecutor;
    private IndexReaderContext indexReaderContext;
    private LeafReaderContext leaf1;
    private LeafReaderContext leaf2;
    private LeafReader leafReader1;
    private LeafReader leafReader2;

    @Mock
    private ClusterService clusterService;

    private NativeEngineKnnVectorQuery objectUnderTest;

    private static ScoreMode scoreMode = ScoreMode.TOP_SCORES;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        openMocks(this);
        objectUnderTest = new NativeEngineKnnVectorQuery(knnQuery, QueryUtils.getInstance(), false);
        reader = createTestIndexReader();
        indexReaderContext = reader.getContext();
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
        when(clusterService.state()).thenReturn(mock(ClusterState.class)); // Mock ClusterState
        // Set ClusterService in KNNSettings
        KNNSettings.state().setClusterService(clusterService);
        when(knnQuery.getQueryVector()).thenReturn(new float[] { 1.0f, 2.0f, 3.0f });  // Example vector

    }

    @SneakyThrows
    public void testMultiLeaf() {
        directory = new ByteBuffersDirectory();
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

        // Initialize DirectoryReader and IndexSearcher
        // Open the real DirectoryReader
        DirectoryReader originalReader = DirectoryReader.open(directory);

        // Define liveDocs for each segment
        Bits liveDocs1 = new Bits() {
            @Override
            public boolean get(int index) {
                return index != 1 && index != 2; // Document 1 and 2 are deleted
            }

            @Override
            public int length() {
                return originalReader.leaves().get(0).reader().maxDoc();
            }
        };

        Bits liveDocs2 = null; // No deletions in the second segment

        // Wrap the DirectoryReader to inject custom liveDocs logic
        directoryReader = CustomFilterDirectoryReader.wrap(originalReader, liveDocs1, liveDocs2);

        // Set the reader and searcher
        reader = directoryReader;
        indexReaderContext = reader.getContext();
        // Extract LeafReaderContext
        List<LeafReaderContext> leaves = reader.leaves();
        assertEquals(2, leaves.size()); // Ensure we have two segments
        leaf1 = leaves.get(0);
        leaf2 = leaves.get(1);
        // Simulate liveDocs for leaf1 (e.g., marking some documents as deleted)
        leafReader1 = leaf1.reader();
        leafReader2 = leaf2.reader();
        // Given
        PerLeafResult leaf1Result = new PerLeafResult(null, buildTopDocs(new HashMap<>(Map.of(0, 1.2f, 1, 5.1f, 2, 2.2f))));
        PerLeafResult leaf2Result = new PerLeafResult(null, buildTopDocs(new HashMap<>(Map.of(4, 3.4f, 3, 5.1f))));

        when(knnWeight.searchLeaf(leaf1, 4)).thenReturn(leaf1Result);
        when(knnWeight.searchLeaf(leaf2, 4)).thenReturn(leaf2Result);
        when(searcher.getIndexReader()).thenReturn(reader);

        // k=4 to make sure we get topk results even if docs are deleted/less in one of the leaves
        when(knnQuery.getK()).thenReturn(4);
        Map<Integer, Float> leaf1ResultLive = Map.of(0, 1.2f);
        TopDocs[] topDocs = {
            ResultUtil.resultMapToTopDocs(leaf1ResultLive, leaf1.docBase),
            ResultUtil.resultMapToTopDocs(convertTopDocsToMap(leaf2Result.getResult()), leaf2.docBase) };
        TopDocs expectedTopDocs = TopDocs.merge(4, topDocs);

        // When
        Weight actual = objectUnderTest.createWeight(searcher, scoreMode, 1);

        // Then
        Query expected = QueryUtils.getInstance().createDocAndScoreQuery(reader, expectedTopDocs);
        assertEquals(expected, actual.getQuery());
    }

    @SneakyThrows
    public void testExplain() {

        List<LeafReaderContext> leaves = reader.leaves();
        assertEquals(1, leaves.size());
        leaf1 = leaves.get(0);
        leafReader1 = leaf1.reader();

        PerLeafResult leafResult = new PerLeafResult(null, buildTopDocs(new HashMap<>(Map.of(4, 3.4f, 3, 5.1f))));

        when(knnWeight.searchLeaf(leaf1, 4)).thenReturn(leafResult);

        Bits liveDocs = mock(Bits.class);
        when(liveDocs.get(anyInt())).thenReturn(true);
        when(liveDocs.get(2)).thenReturn(false);
        when(liveDocs.get(1)).thenReturn(false);

        // k=4 to make sure we get topk results even if docs are deleted/less in one of the leaves
        when(knnQuery.getK()).thenReturn(4);

        TopDocs[] topDocs = { leafResult.getResult() };
        TopDocs expectedTopDocs = TopDocs.merge(4, topDocs);

        // When
        Weight actual = objectUnderTest.createWeight(searcher, scoreMode, 1);

        // Then
        Query expected = QueryUtils.getInstance().createDocAndScoreQuery(reader, expectedTopDocs);
        assertEquals(expected, actual.getQuery());
        for (ScoreDoc scoreDoc : expectedTopDocs.scoreDocs) {
            int docId = scoreDoc.doc;
            if (docId == 0) continue;
            float score = scoreDoc.score;
            actual.explain(leaf1, docId);
            verify(knnWeight).explain(leaf1, docId, score);
        }
    }

    @SneakyThrows
    public void testRescoreWhenShardLevelRescoringEnabled() {

        directory = new ByteBuffersDirectory();
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
        Bits liveDocs1 = null;

        Bits liveDocs2 = null;

        DirectoryReader originalReader = DirectoryReader.open(directory);

        // Wrap the DirectoryReader to inject custom liveDocs logic
        directoryReader = CustomFilterDirectoryReader.wrap(originalReader, liveDocs1, liveDocs2);

        // Set the reader and searcher
        reader = directoryReader;
        indexReaderContext = reader.getContext();
        // Extract LeafReaderContext
        List<LeafReaderContext> leaves = reader.leaves();
        assertEquals(2, leaves.size()); // Ensure we have two segments
        leaf1 = leaves.get(0);
        leaf2 = leaves.get(1);
        // Simulate liveDocs for leaf1 (e.g., marking some documents as deleted)
        leafReader1 = leaf1.reader();
        leafReader2 = leaf2.reader();

        int k = 2;
        PerLeafResult initialLeaf1Results = new PerLeafResult(null, buildTopDocs(new HashMap<>(Map.of(0, 21f, 1, 19f, 2, 17f))));
        PerLeafResult initialLeaf2Results = new PerLeafResult(null, buildTopDocs(new HashMap<>(Map.of(0, 20f, 1, 18f, 2, 16f))));
        Map<Integer, Float> rescoredLeaf1Results = new HashMap<>(Map.of(0, 18f, 1, 20f));
        Map<Integer, Float> rescoredLeaf2Results = new HashMap<>(Map.of(0, 21f));

        RescoreContext rescoreContext = RescoreContext.builder().oversampleFactor(1.5f).build();
        int firstPassK = rescoreContext.getFirstPassK(k, true, 1);
        when(knnQuery.getRescoreContext()).thenReturn(RescoreContext.builder().oversampleFactor(1.5f).build());
        when(knnQuery.getK()).thenReturn(k);
        when(knnWeight.getQuery()).thenReturn(knnQuery);
        when(knnWeight.searchLeaf(leaf1, firstPassK)).thenReturn(initialLeaf1Results);
        when(knnWeight.searchLeaf(leaf2, firstPassK)).thenReturn(initialLeaf2Results);
        when(knnWeight.exactSearch(eq(leaf1), any())).thenReturn(buildTopDocs(rescoredLeaf1Results));
        when(knnWeight.exactSearch(eq(leaf2), any())).thenReturn(buildTopDocs(rescoredLeaf2Results));
        when(searcher.getIndexReader()).thenReturn(reader);

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
        PerLeafResult leaf1Result = new PerLeafResult(null, buildTopDocs(new HashMap<>(Map.of(0, 1.2f, 1, 5.1f, 2, 2.2f))));
        List<LeafReaderContext> leaves = reader.leaves();
        leaf1 = leaves.get(0);
        when(knnWeight.searchLeaf(leaf1, k)).thenReturn(leaf1Result);
        when(knnQuery.getK()).thenReturn(k);
        TopDocs expectedTopDocs = ResultUtil.resultMapToTopDocs(convertTopDocsToMap(leaf1Result.getResult()), leaf1.docBase);

        // When
        Weight actual = objectUnderTest.createWeight(searcher, scoreMode, boost);

        // Then
        Query expected = QueryUtils.getInstance().createDocAndScoreQuery(reader, expectedTopDocs);
        assertEquals(expected, actual.getQuery());
    }

    @SneakyThrows
    public void testNoMatch() {
        // Given
        List<LeafReaderContext> leaves = reader.leaves();
        leaf1 = leaves.get(0);
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
        directory = new ByteBuffersDirectory();
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
        Bits liveDocs1 = null;

        Bits liveDocs2 = null;

        DirectoryReader originalReader = DirectoryReader.open(directory);

        // Wrap the DirectoryReader to inject custom liveDocs logic
        directoryReader = CustomFilterDirectoryReader.wrap(originalReader, liveDocs1, liveDocs2);

        // Set the reader and searcher
        reader = directoryReader;
        indexReaderContext = reader.getContext();
        // Extract LeafReaderContext
        List<LeafReaderContext> leaves = reader.leaves();
        assertEquals(2, leaves.size()); // Ensure we have two segments
        leaf1 = leaves.get(0);
        leaf2 = leaves.get(1);
        // Simulate liveDocs for leaf1 (e.g., marking some documents as deleted)
        leafReader1 = leaf1.reader();
        leafReader2 = leaf2.reader();

        int k = 2;
        int firstPassK = 100;
        PerLeafResult initialLeaf1Results = new PerLeafResult(null, buildTopDocs(new HashMap<>(Map.of(0, 21f, 1, 19f, 2, 17f, 3, 15f))));
        PerLeafResult initialLeaf2Results = new PerLeafResult(null, buildTopDocs(new HashMap<>(Map.of(0, 20f, 1, 18f, 2, 16f, 3, 14f))));
        TopDocs topDocs1 = ResultUtil.resultMapToTopDocs(Map.of(0, 18f, 1, 20f), 0);
        TopDocs topDocs2 = ResultUtil.resultMapToTopDocs(Map.of(0, 21f), 4);
        when(knnQuery.getRescoreContext()).thenReturn(RescoreContext.builder().oversampleFactor(1.5f).build());
        when(knnQuery.getK()).thenReturn(k);
        when(knnWeight.getQuery()).thenReturn(knnQuery);
        when(knnWeight.searchLeaf(leaf1, firstPassK)).thenReturn(initialLeaf1Results);
        when(knnWeight.searchLeaf(leaf2, firstPassK)).thenReturn(initialLeaf2Results);

        when(knnWeight.exactSearch(eq(leaf1), any())).thenReturn(topDocs1);
        when(knnWeight.exactSearch(eq(leaf2), any())).thenReturn(topDocs2);
        when(searcher.getIndexReader()).thenReturn(reader);

        try (
            MockedStatic<KNNSettings> mockedKnnSettings = mockStatic(KNNSettings.class);
            MockedStatic<ResultUtil> mockedResultUtil = mockStatic(ResultUtil.class)
        ) {

            // When shard-level re-scoring is enabled
            mockedKnnSettings.when(() -> KNNSettings.isShardLevelRescoringDisabledForDiskBasedVector(any())).thenReturn(false);

            mockedResultUtil.when(() -> ResultUtil.reduceToTopK(any(), anyInt())).thenAnswer(InvocationOnMock::callRealMethod);
            mockedResultUtil.when(() -> ResultUtil.resultMapToDocIds(any(), anyInt())).thenAnswer(InvocationOnMock::callRealMethod);

            // Run
            Weight actual = objectUnderTest.createWeight(searcher, scoreMode, 1);

            // Verify
            TopDocs[] topDocs = { topDocs1, topDocs2 };
            TopDocs expectedTopDocs = TopDocs.merge(k, topDocs);
            Query expected = QueryUtils.getInstance().createDocAndScoreQuery(reader, expectedTopDocs);
            assertEquals(expected, actual.getQuery());
        }
    }

    @SneakyThrows
    public void testExpandNestedDocs() {
        directory = new ByteBuffersDirectory();
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
        Bits liveDocs1 = null;

        Bits liveDocs2 = null;

        DirectoryReader originalReader = DirectoryReader.open(directory);

        // Wrap the DirectoryReader to inject custom liveDocs logic
        directoryReader = CustomFilterDirectoryReader.wrap(originalReader, liveDocs1, liveDocs2);

        // Set the reader and searcher
        reader = directoryReader;
        indexReaderContext = reader.getContext();
        // Extract LeafReaderContext
        List<LeafReaderContext> leaves = reader.leaves();
        assertEquals(2, leaves.size()); // Ensure we have two segments
        leaf1 = leaves.get(0);
        leaf2 = leaves.get(1);
        // Simulate liveDocs for leaf1 (e.g., marking some documents as deleted)
        leafReader1 = leaf1.reader();
        leafReader2 = leaf2.reader();
        Bits queryFilterBits = mock(Bits.class);
        HashMap<Integer, Float> leaf1Result = new HashMap<>(Map.of(0, 19f, 1, 20f, 2, 17f, 3, 15f));
        PerLeafResult initialLeaf1Results = new PerLeafResult(queryFilterBits, buildTopDocs(leaf1Result));
        HashMap<Integer, Float> leaf2Result = new HashMap<>(Map.of(0, 21f, 1, 18f, 2, 16f, 3, 14f));
        PerLeafResult initialLeaf2Results = new PerLeafResult(queryFilterBits, buildTopDocs(leaf2Result));

        Map<Integer, Float> exactSearchLeaf1Result = new HashMap<>(Map.of(1, 20f));
        Map<Integer, Float> exactSearchLeaf2Result = new HashMap<>(Map.of(0, 21f));
        List<Map<Integer, Float>> perLeafResults = Arrays.asList(exactSearchLeaf1Result, exactSearchLeaf2Result);

        TopDocs topDocs1 = ResultUtil.resultMapToTopDocs(exactSearchLeaf1Result, leaf1.docBase);
        TopDocs topDocs2 = ResultUtil.resultMapToTopDocs(exactSearchLeaf2Result, leaf2.docBase);
        TopDocs topK = TopDocs.merge(2, new TopDocs[] { topDocs1, topDocs2 });
        when(searcher.getIndexReader()).thenReturn(reader);

        int k = 2;
        when(knnQuery.getRescoreContext()).thenReturn(null);
        when(knnQuery.getK()).thenReturn(k);

        BitSetProducer parentFilter = mock(BitSetProducer.class);
        when(knnQuery.getParentsFilter()).thenReturn(parentFilter);
        when(knnWeight.searchLeaf(leaf1, k)).thenReturn(initialLeaf1Results);
        when(knnWeight.searchLeaf(leaf2, k)).thenReturn(initialLeaf2Results);
        when(knnWeight.exactSearch(eq(leaf1), any())).thenReturn(buildTopDocs(exactSearchLeaf1Result));
        when(knnWeight.exactSearch(eq(leaf2), any())).thenReturn(buildTopDocs(exactSearchLeaf2Result));
        Weight filterWeight = mock(Weight.class);
        when(knnWeight.getFilterWeight()).thenReturn(filterWeight);

        DocIdSetIterator allSiblings = mock(DocIdSetIterator.class);
        when(allSiblings.nextDoc()).thenReturn(1, 2, DocIdSetIterator.NO_MORE_DOCS);

        Weight expectedWeight = mock(Weight.class);
        Query finalQuery = mock(Query.class);
        when(finalQuery.createWeight(searcher, scoreMode, 1)).thenReturn(expectedWeight);

        QueryUtils queryUtils = mock(QueryUtils.class);
        when(queryUtils.getAllSiblings(any(), any(), any(), any())).thenReturn(allSiblings);
        when(queryUtils.createDocAndScoreQuery(eq(reader), any(), eq(knnWeight))).thenReturn(finalQuery);

        // Run
        NativeEngineKnnVectorQuery query = new NativeEngineKnnVectorQuery(knnQuery, queryUtils, true);
        Weight finalWeigh = query.createWeight(searcher, scoreMode, 1.f);

        // Verify
        assertEquals(expectedWeight, finalWeigh);
        verify(queryUtils).getAllSiblings(leaf1, perLeafResults.get(0).keySet(), parentFilter, queryFilterBits);
        verify(queryUtils).getAllSiblings(leaf2, perLeafResults.get(1).keySet(), parentFilter, queryFilterBits);
        ArgumentCaptor<TopDocs> topDocsCaptor = ArgumentCaptor.forClass(TopDocs.class);
        verify(queryUtils).createDocAndScoreQuery(eq(reader), topDocsCaptor.capture(), eq(knnWeight));
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

    private IndexReader createTestIndexReader() throws IOException {
        ByteBuffersDirectory directory = new ByteBuffersDirectory();
        IndexWriter writer = new IndexWriter(directory, new IndexWriterConfig(new MockAnalyzer(random())));
        writer.addDocument(new Document());
        writer.close();
        return DirectoryReader.open(directory);
    }
}

class CustomFilterDirectoryReader extends FilterDirectoryReader {

    private final Bits liveDocs1;
    private final Bits liveDocs2;

    protected CustomFilterDirectoryReader(DirectoryReader in, Bits liveDocs1, Bits liveDocs2) throws IOException {
        super(in, getWrapper(liveDocs1, liveDocs2));
        this.liveDocs1 = liveDocs1;
        this.liveDocs2 = liveDocs2;
    }

    @Override
    protected DirectoryReader doWrapDirectoryReader(DirectoryReader in) throws IOException {
        return new CustomFilterDirectoryReader(in, liveDocs1, liveDocs2);
    }

    private static SubReaderWrapper getWrapper(Bits liveDocs1, Bits liveDocs2) {
        return new SubReaderWrapper() {
            @Override
            public LeafReader wrap(LeafReader reader) {
                if (reader.getContext().ord == 0) { // First segment
                    return new FilterLeafReader(reader) {
                        /**
                         * @return
                         */
                        @Override
                        public CacheHelper getReaderCacheHelper() {
                            return null;
                        }

                        /**
                         * @return
                         */
                        @Override
                        public CacheHelper getCoreCacheHelper() {
                            return null;
                        }

                        @Override
                        public Bits getLiveDocs() {
                            return liveDocs1;
                        }
                    };
                } else if (reader.getContext().ord == 1) { // Second segment
                    return new FilterLeafReader(reader) {
                        /**
                         * @return
                         */
                        @Override
                        public CacheHelper getReaderCacheHelper() {
                            return null;
                        }

                        /**
                         * @return
                         */
                        @Override
                        public CacheHelper getCoreCacheHelper() {
                            return null;
                        }

                        @Override
                        public Bits getLiveDocs() {
                            return liveDocs2;
                        }
                    };
                } else {
                    return reader; // Default case
                }
            }
        };
    }

    // Remove the static modifier to fix the error
    public static DirectoryReader wrap(DirectoryReader reader, Bits liveDocs1, Bits liveDocs2) throws IOException {
        return new CustomFilterDirectoryReader(reader, liveDocs1, liveDocs2);
    }

    /**
     * @return
     */
    @Override
    public CacheHelper getReaderCacheHelper() {
        return null;
    }
}
