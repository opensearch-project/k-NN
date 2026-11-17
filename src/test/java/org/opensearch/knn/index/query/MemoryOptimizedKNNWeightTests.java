/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field.Store;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.QueryTimeout;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.TimeLimitingKnnCollectorManager;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.mockito.MockedStatic;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNNCodecVersion;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.memoryoptsearch.MemoryOptimizedKNNWeight;
import org.opensearch.knn.index.query.memoryoptsearch.RadiusVectorSimilarityCollector;
import org.opensearch.lucene.ReentrantKnnCollectorManager;

import java.lang.reflect.Field;
import java.nio.file.Path;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.DEFAULT_LUCENE_RADIAL_SEARCH_DECAY;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;

public class MemoryOptimizedKNNWeightTests extends KNNTestCase {

    private static final int K = 10;
    private static final int FILTERED_DOCS = 100_000;
    private static final String FIELD_NAME = "test-field";
    private static final String INDEX_NAME = "test-index";
    private static final float[] QUERY_VECTOR = new float[] { 1.0f, 2.0f };
    private static final float RADIUS = 0.5f;
    // More docs than K so that exact search isn't chosen up front, fewer than the segment so that the filter isn't dropped.
    private static final int FILTER_CARDINALITY = 50;
    private static final int SEGMENT_MAX_DOC = 100;

    private MockedStatic<KNNSettings> knnSettings;
    private Directory directory;
    private DirectoryReader reader;
    private LeafReaderContext leafContext;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        knnSettings = mockStatic(KNNSettings.class);
        knnSettings.when(() -> KNNSettings.getFilteredExactSearchThreshold(any()))
            .thenReturn(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD_DEFAULT_VALUE);
        knnSettings.when(() -> KNNSettings.isKnnIndexFaissEfficientFilterExactSearchDisabled(any())).thenReturn(false);

        // Include the queried field so visibility checks pass. ANN, exact search and the filter are stubbed on the
        // weight, so the real one-segment index needs no vectors.
        directory = newDirectory();
        try (IndexWriter writer = new IndexWriter(directory, new IndexWriterConfig())) {
            final Document document = new Document();
            document.add(new StringField(FIELD_NAME, "value", Store.NO));
            writer.addDocument(document);
        }
        reader = DirectoryReader.open(directory);
        leafContext = reader.leaves().get(0);
    }

    @Override
    public void tearDown() throws Exception {
        reader.close();
        directory.close();
        knnSettings.close();
        super.tearDown();
    }

    public void testAcornIsDisabled() throws Exception {
        Field field = MemoryOptimizedKNNWeight.class.getDeclaredField("DEFAULT_HNSW_SEARCH_STRATEGY");
        field.setAccessible(true);
        KnnSearchStrategy.Hnsw strategy = (KnnSearchStrategy.Hnsw) field.get(null);

        assertEquals(
            "ACORN threshold must be 0 to disable filtered search and match Lucene 10.4 behavior",
            0,
            strategy.filteredSearchThreshold()
        );
        assertFalse("useFilteredSearch must return false for any filtering rate when threshold is 0", strategy.useFilteredSearch(0.5f));
    }

    // Validates that the memory-optimized search (MOS) radius path builds the decay-based
    // RadiusVectorSimilarityCollector and wires in the shared decay factor
    // (DEFAULT_LUCENE_RADIAL_SEARCH_DECAY = 0.95). A radius (r-NN) query is simulated by passing k = null,
    // which routes the constructor to the radius branch that builds the collector manager.
    public void testRadiusSearch_usesDecayBasedCollector() throws Exception {
        final KNNQuery knnQuery = mock(KNNQuery.class);
        when(knnQuery.getRadius()).thenReturn(0.5f);
        final IndexSearcher searcher = mock(IndexSearcher.class);

        // k == null -> radius search branch, which creates the RadiusVectorSimilarityCollector manager.
        final MemoryOptimizedKNNWeight weight = new MemoryOptimizedKNNWeight(knnQuery, 1.0f, null, searcher, null);

        // Reach the private collector manager built for the radius search path.
        final KnnCollectorManager manager = getCollectorManager(weight);

        // The radius lambda ignores the search strategy and context, so null is acceptable here.
        final KnnCollector collector = manager.newCollector(Integer.MAX_VALUE, null, null);
        assertTrue(
            "MOS radius search must use the decay-based RadiusVectorSimilarityCollector",
            collector instanceof RadiusVectorSimilarityCollector
        );

        // Verify the decay factor wired into the collector is the shared default (0.95).
        final Field decayField = RadiusVectorSimilarityCollector.class.getDeclaredField("decay");
        decayField.setAccessible(true);
        assertEquals(DEFAULT_LUCENE_RADIAL_SEARCH_DECAY, (float) decayField.get(collector), 0.0f);
    }

    public void testAnnCollectorManager_whenSearcherHasTimeout_thenIsTimeLimited() throws Exception {
        final QueryTimeout timeout = () -> false;
        final IndexSearcher searcher = mock(IndexSearcher.class);
        when(searcher.getTimeout()).thenReturn(timeout);

        for (BitSetProducer parentsFilter : new BitSetProducer[] { null, mock(BitSetProducer.class) }) {
            final KNNQuery knnQuery = mock(KNNQuery.class);
            when(knnQuery.getParentsFilter()).thenReturn(parentsFilter);

            final MemoryOptimizedKNNWeight weight = new MemoryOptimizedKNNWeight(knnQuery, 1.0f, null, searcher, 10);

            final KnnCollectorManager manager = getCollectorManager(weight);
            assertTrue(
                "expected a time-limited collector manager, got " + manager.getClass(),
                manager instanceof TimeLimitingKnnCollectorManager
            );
            assertSame(timeout, ((TimeLimitingKnnCollectorManager) manager).getQueryTimeout());
            assertSame(timeout, ((KNNWeight) weight).getQueryTimeout());
        }
    }

    public void testRadiusCollector_whenTimeoutFires_thenEarlyTerminates() throws Exception {
        final AtomicBoolean timedOut = new AtomicBoolean(false);
        final KNNQuery knnQuery = mock(KNNQuery.class);
        when(knnQuery.getRadius()).thenReturn(0.5f);
        final IndexSearcher searcher = mock(IndexSearcher.class);
        when(searcher.getTimeout()).thenReturn(timedOut::get);

        final MemoryOptimizedKNNWeight weight = new MemoryOptimizedKNNWeight(knnQuery, 1.0f, null, searcher, null);
        final KnnCollector collector = getCollectorManager(weight).newCollector(Integer.MAX_VALUE, null, null);

        assertFalse(collector.earlyTerminated());
        timedOut.set(true);
        assertTrue("the collector must stop once the query times out", collector.earlyTerminated());
    }

    public void testGetQueryTimeout_whenSearcherHasNoTimeout_thenNull() {
        final KNNQuery knnQuery = mock(KNNQuery.class);
        final MemoryOptimizedKNNWeight weight = new MemoryOptimizedKNNWeight(knnQuery, 1.0f, null, mock(IndexSearcher.class), 10);

        assertNull(((KNNWeight) weight).getQueryTimeout());
    }

    public void testSearchLeaf_whenTimedOutAfterPartialAnnResults_thenSkipsExactSearchFallback() throws Exception {
        final TopDocs partialAnnResults = new TopDocs(
            new TotalHits(1, TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO),
            new ScoreDoc[] { new ScoreDoc(0, 1.0f) }
        );
        final MemoryOptimizedKNNWeight weight = filteredWeightReturningAnnResults(() -> true, partialAnnResults);

        final PerLeafResult result = weight.searchLeaf(leafContext, K);

        verify(weight, never()).exactSearch(any(), any());
        assertSame(partialAnnResults, result.getResult());
        assertEquals(PerLeafResult.SearchMode.APPROXIMATE_SEARCH, result.getSearchMode());
    }

    public void testSearchLeaf_whenNotTimedOutWithPartialAnnResults_thenFallsBackToExactSearch() throws Exception {
        final TopDocs partialAnnResults = new TopDocs(
            new TotalHits(1, TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO),
            new ScoreDoc[] { new ScoreDoc(0, 1.0f) }
        );
        final MemoryOptimizedKNNWeight weight = filteredWeightReturningAnnResults(() -> false, partialAnnResults);

        final PerLeafResult result = weight.searchLeaf(leafContext, K);

        verify(weight, times(1)).exactSearch(any(), any());
        assertEquals(PerLeafResult.SearchMode.EXACT_SEARCH, result.getSearchMode());
    }

    // Filtered radial search should fall back to exact search when it receives the exhausted search budget signal. The signal
    // must survive even when the budget ran out before collecting any hit.
    public void testSearchLeaf_whenFilteredRadialSearchExhaustsBudgetWithoutHits_thenFallsBackToExactSearch() throws Exception {
        final KNNQuery knnQuery = KNNQuery.builder()
            .field(FIELD_NAME)
            .queryVector(QUERY_VECTOR)
            .radius(RADIUS)
            .indexName(INDEX_NAME)
            .vectorDataType(VectorDataType.FLOAT)
            .isMemoryOptimizedSearch(true)
            .build();
        final LeafReaderContext context = mockLeafWhoseVectorSearchCollectsNothing(true);
        final MemoryOptimizedKNNWeight weight = filteredWeightSearchingMockedLeaf(knnQuery, null);

        final PerLeafResult result = weight.searchLeaf(context, 0);

        verify(weight, times(1)).exactSearch(any(), any());
        assertEquals(PerLeafResult.SearchMode.EXACT_SEARCH, result.getSearchMode());
    }

    public void testSearchLeaf_whenFilteredTopKSearchExhaustsBudgetWithoutHits_thenExplainsExhaustedSearch() throws Exception {
        final KNNQuery knnQuery = topKQuery(true);
        final LeafReaderContext context = mockLeafWhoseVectorSearchCollectsNothing(true);
        final MemoryOptimizedKNNWeight weight = filteredWeightSearchingMockedLeaf(knnQuery, K);

        final PerLeafResult result = weight.searchLeaf(context, K);

        verify(weight, times(1)).exactSearch(any(), any());
        assertEquals(PerLeafResult.SearchMode.EXACT_SEARCH, result.getSearchMode());
        assertEquals(Integer.valueOf(0), ((KNNWeight) weight).getKnnExplanation().getAnnResult(context.id()));
        assertEquals(
            "explain must report that the search budget was exhausted, even though no hit was collected",
            Boolean.TRUE,
            ((KNNWeight) weight).getKnnExplanation().getExhaustedSearch(context.id())
        );
    }

    public void testApproximateSearch_whenBudgetExhaustedWithoutHits_thenKeepsRelation() throws Exception {
        final LeafReaderContext context = mockLeafWhoseVectorSearchCollectsNothing(true);
        final MemoryOptimizedKNNWeight weight = filteredWeightSearchingMockedLeaf(topKQuery(false), K);

        final TopDocs topDocs = weight.approximateSearch(context, partialSegmentFilter(), FILTER_CARDINALITY, K);

        assertEquals(0, topDocs.scoreDocs.length);
        assertEquals(0, topDocs.totalHits.value());
        assertEquals(TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO, topDocs.totalHits.relation());
    }

    public void testApproximateSearch_whenNoHitsAndBudgetNotExhausted_thenReturnsEmptyTopDocs() throws Exception {
        final LeafReaderContext context = mockLeafWhoseVectorSearchCollectsNothing(false);
        final MemoryOptimizedKNNWeight weight = filteredWeightSearchingMockedLeaf(topKQuery(false), K);

        final TopDocs topDocs = weight.approximateSearch(context, partialSegmentFilter(), FILTER_CARDINALITY, K);

        assertSame(KNNWeight.EMPTY_TOPDOCS, topDocs);
    }

    private static KNNQuery topKQuery(final boolean explain) {
        return KNNQuery.builder()
            .field(FIELD_NAME)
            .queryVector(QUERY_VECTOR)
            .k(K)
            .indexName(INDEX_NAME)
            .vectorDataType(VectorDataType.FLOAT)
            .isMemoryOptimizedSearch(true)
            .explain(explain)
            .build();
    }

    private static FixedBitSet partialSegmentFilter() {
        final FixedBitSet filter = new FixedBitSet(SEGMENT_MAX_DOC);
        filter.set(0, FILTER_CARDINALITY);
        return filter;
    }

    /**
     * A filtered MOS weight whose approximate search is NOT stubbed, so that results go through both
     * {@code MemoryOptimizedKNNWeight#queryIndex} and {@link KNNWeight#approximateSearch}. Only the filter and exact search are
     * stubbed. It should be used with a leaf created by {@link #mockLeafWhoseVectorSearchCollectsNothing}.
     *
     * @param k K for top-k search, or null for radial search.
     */
    private MemoryOptimizedKNNWeight filteredWeightSearchingMockedLeaf(final KNNQuery knnQuery, final Integer k) throws Exception {
        // Send the filter to ANN search.
        knnSettings.when(() -> KNNSettings.getFilteredExactSearchThreshold(any())).thenReturn(0);

        final MemoryOptimizedKNNWeight weight = spy(
            new MemoryOptimizedKNNWeight(knnQuery, 1.0f, mock(Weight.class), mock(IndexSearcher.class), k)
        );
        doReturn(partialSegmentFilter()).when(weight).getFilteredDocsBitSet(any());
        doReturn(new TopDocs(new TotalHits(0, TotalHits.Relation.EQUAL_TO), new ScoreDoc[0])).when(weight).exactSearch(any(), any());

        if (k != null) {
            // Provide plain top-k collectors through the re-entrant collector manager since the mocked leaf has no parent reader.
            final ReentrantKnnCollectorManager collectorManager = mock(ReentrantKnnCollectorManager.class);
            when(collectorManager.newCollector(anyInt(), any(), any())).thenAnswer(
                invocation -> new TopKnnCollector(k, invocation.getArgument(0), invocation.getArgument(1))
            );
            weight.setReentrantKNNCollectorManager(collectorManager);
        }
        return weight;
    }

    /**
     * A mocked segment having a Faiss engine file for the field, whose vector search never collects a hit.
     *
     * @param exhaustBudget If true, the vector search visits as many vectors as the collector's visit limit, which is what happens
     *                      when the search budget runs out before reaching any doc accepted by the filter.
     */
    private static LeafReaderContext mockLeafWhoseVectorSearchCollectsNothing(final boolean exhaustBudget) throws Exception {
        final FSDirectory fsDirectory = mock(FSDirectory.class);
        when(fsDirectory.getDirectory()).thenReturn(mock(Path.class));
        final SegmentInfo segmentInfo = new SegmentInfo(
            fsDirectory,
            Version.LATEST,
            Version.LATEST,
            "_0",
            SEGMENT_MAX_DOC,
            false,
            false,
            KNNCodecVersion.CURRENT_DEFAULT,
            Map.of(),
            new byte[StringHelper.ID_LENGTH],
            Map.of(),
            Sort.RELEVANCE
        );
        segmentInfo.setFiles(Set.of("_0_165_" + FIELD_NAME + KNNEngine.FAISS.getExtension()));

        final FieldInfo fieldInfo = mock(FieldInfo.class);
        final Map<String, String> attributes = Map.of(KNN_ENGINE, KNNEngine.FAISS.getName(), SPACE_TYPE, SpaceType.L2.getValue());
        when(fieldInfo.attributes()).thenReturn(attributes);
        when(fieldInfo.getAttribute(any())).thenAnswer(invocation -> attributes.get(invocation.<String>getArgument(0)));
        final FieldInfos fieldInfos = mock(FieldInfos.class);
        when(fieldInfos.fieldInfo(FIELD_NAME)).thenReturn(fieldInfo);

        final KnnVectorsReader vectorsReader = mock(KnnVectorsReader.class);
        doAnswer(invocation -> {
            if (exhaustBudget) {
                final KnnCollector collector = invocation.getArgument(2);
                collector.incVisitedCount(Math.toIntExact(collector.visitLimit()));
            }
            return null;
        }).when(vectorsReader).search(any(), any(float[].class), any(KnnCollector.class), any());

        final SegmentReader segmentReader = mock(SegmentReader.class);
        when(segmentReader.directory()).thenReturn(fsDirectory);
        when(segmentReader.getSegmentInfo()).thenReturn(
            new SegmentCommitInfo(segmentInfo, 0, 0, 0, 0, 0, new byte[StringHelper.ID_LENGTH])
        );
        when(segmentReader.maxDoc()).thenReturn(SEGMENT_MAX_DOC);
        when(segmentReader.getFieldInfos()).thenReturn(fieldInfos);
        when(segmentReader.getVectorReader()).thenReturn(vectorsReader);

        final LeafReaderContext context = mock(LeafReaderContext.class);
        when(context.reader()).thenReturn(segmentReader);
        when(context.id()).thenReturn(new Object());
        return context;
    }

    /**
     * A filtered MOS weight whose ANN search returns {@code annResults} and whose filter matches enough docs that
     * exact search isn't chosen up front, so only the after-ANN fallback can trigger exact search.
     */
    private MemoryOptimizedKNNWeight filteredWeightReturningAnnResults(final QueryTimeout timeout, final TopDocs annResults)
        throws Exception {
        final KNNQuery knnQuery = mock(KNNQuery.class);
        when(knnQuery.getK()).thenReturn(K);
        when(knnQuery.getQueryDimension()).thenReturn(768);
        when(knnQuery.getIndexName()).thenReturn("test-index");
        when(knnQuery.getField()).thenReturn(FIELD_NAME);
        final IndexSearcher searcher = mock(IndexSearcher.class);
        when(searcher.getTimeout()).thenReturn(timeout);

        final MemoryOptimizedKNNWeight weight = spy(new MemoryOptimizedKNNWeight(knnQuery, 1.0f, mock(Weight.class), searcher, K));
        final FixedBitSet filter = new FixedBitSet(FILTERED_DOCS);
        filter.set(0, FILTERED_DOCS);
        doReturn(filter).when(weight).getFilteredDocsBitSet(any());
        doReturn(annResults).when(weight).approximateSearch(any(), any(), anyInt(), anyInt());
        doReturn(new TopDocs(new TotalHits(0, TotalHits.Relation.EQUAL_TO), new ScoreDoc[0])).when(weight).exactSearch(any(), any());
        return weight;
    }

    private static KnnCollectorManager getCollectorManager(final MemoryOptimizedKNNWeight weight) throws Exception {
        final Field managerField = MemoryOptimizedKNNWeight.class.getDeclaredField("knnCollectorManager");
        managerField.setAccessible(true);
        return (KnnCollectorManager) managerField.get(weight);
    }
}
