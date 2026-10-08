/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.QueryTimeout;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TimeLimitingKnnCollectorManager;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.FixedBitSet;
import org.mockito.MockedStatic;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.query.memoryoptsearch.MemoryOptimizedKNNWeight;
import org.opensearch.knn.index.query.memoryoptsearch.RadiusVectorSimilarityCollector;

import java.lang.reflect.Field;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.DEFAULT_LUCENE_RADIAL_SEARCH_DECAY;

public class MemoryOptimizedKNNWeightTests extends KNNTestCase {

    private static final int K = 10;
    private static final int FILTERED_DOCS = 100_000;

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

        // A real one-segment index, so searchLeaf sees a genuine SegmentReader. ANN, exact search and the filter are
        // stubbed on the weight, so the segment needs no vectors.
        directory = newDirectory();
        try (IndexWriter writer = new IndexWriter(directory, new IndexWriterConfig())) {
            writer.addDocument(new Document());
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
        when(knnQuery.getField()).thenReturn("test-field");
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
