/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.index.CompositeReaderContext;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.Weight;
import org.junit.Before;
import org.junit.Test;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.query.KNNQuery;
import org.opensearch.knn.index.query.PerLeafResult;
import org.opensearch.knn.index.query.common.QueryUtils;
import org.opensearch.knn.index.query.memoryoptsearch.MemoryOptimizedKNNWeight;
import org.opensearch.knn.index.query.nativelib.NativeEngineKnnVectorQuery;

import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;

import static org.junit.Assert.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyFloat;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class OptimisticSearchTests {
    private static final int DEFAULT_K = 50;

    private IndexSearcher searcher;
    private LeafReader parentIndexReader;
    private MemoryOptimizedKNNWeight knnWeight;
    private KNNQuery knnQuery;

    @Before
    @SneakyThrows
    public void setup() {
        parentIndexReader = mock(LeafReader.class);
        final LeafReaderContext indexReaderContext = mock(LeafReaderContext.class);
        when(indexReaderContext.id()).thenReturn(this);
        when(parentIndexReader.getContext()).thenReturn(indexReaderContext);

        searcher = mock(IndexSearcher.class);
        when(searcher.getIndexReader()).thenReturn(parentIndexReader);
        final TaskExecutor executor = new TaskExecutor(Executors.newSingleThreadExecutor());
        when(searcher.getTaskExecutor()).thenReturn(executor);

        knnWeight = mock(MemoryOptimizedKNNWeight.class);

        knnQuery = mock(KNNQuery.class);
        when(knnQuery.createWeight(any(), any(), anyFloat())).thenReturn(knnWeight);
        when(knnQuery.getK()).thenReturn(DEFAULT_K);
        when(knnQuery.isMemoryOptimizedSearch()).thenReturn(true);
        when(knnQuery.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        when(knnQuery.getQueryVector()).thenReturn(new float[8]);
        when(knnQuery.getField()).thenReturn("field");
    }

    @Test
    @SneakyThrows
    public void testOptimisticSearchWith5Segments2Reentering() {
        // 5 segments, only 2 segments returned results whose min score >= kth largest score
        testOptimisticSearch(5, 2, true);
        testOptimisticSearch(5, 2, false);
    }

    @Test
    @SneakyThrows
    public void testOptimisticSearchWith5Segments0Reentering() {
        // 5 segments, none returns results whose min score >= kth largest score
        testOptimisticSearch(5, 0, true);
    }

    @Test
    @SneakyThrows
    public void testOptimisticSearchWith0Segments() {
        // Empty case
        testOptimisticSearch(0, 0, true);
    }

    @Test
    @SneakyThrows
    public void testOptimisticSearchWith1Segments() {
        // There's only single segment, optimistic search should be disabled.
        testOptimisticSearch(1, 0, true);
    }

    @SneakyThrows
    private void testOptimisticSearch(final int numSegments, final int numSegmentsForReentering, final boolean isApproximateSearch) {
        // Create a query
        final NativeEngineKnnVectorQuery query = new NativeEngineKnnVectorQuery(knnQuery, QueryUtils.getInstance(), false);

        // Create answer sets for 1st phase search
        final List<List<ScoreDoc>> searchResults = new ArrayList<>();
        for (int i = 0; i < numSegments; i++) {
            searchResults.add(new ArrayList<>());
        }

        // Score distribution = 0.123, 1.123, ..., (#segments * k - 1) + 0.123
        final float kthLargestScore = (numSegments * DEFAULT_K - DEFAULT_K) + 0.123F;
        for (int i = 0, j = 0; i < numSegments * DEFAULT_K; j = (j + 1) % numSegments) {
            final float score = i + 0.123F;
            final List<ScoreDoc> scoreDocs = searchResults.get(j);
            int prevDocId = -1;
            if (scoreDocs.isEmpty() == false) {
                prevDocId = scoreDocs.get(scoreDocs.size() - 1).doc;
            }
            if (j >= numSegmentsForReentering || score >= kthLargestScore) {
                scoreDocs.add(new ScoreDoc(prevDocId + 1, score));
                ++i;
            }
        }

        // Sort by score by desc
        for (List<ScoreDoc> scoreDocs : searchResults) {
            scoreDocs.sort((a, b) -> Float.compare(b.score, a.score));
        }

        // Wrap results with PerLeafResult
        final List<PerLeafResult> perLeafResults = new ArrayList<>();
        for (int i = 0; i < numSegments; i++) {
            perLeafResults.add(
                new PerLeafResult(
                    null,
                    0,
                    new TopDocs(
                        new TotalHits(searchResults.get(i).size(), TotalHits.Relation.EQUAL_TO),
                        searchResults.get(i).toArray(new ScoreDoc[0])
                    ),
                    isApproximateSearch ? PerLeafResult.SearchMode.APPROXIMATE_SEARCH : PerLeafResult.SearchMode.EXACT_SEARCH
                )
            );
        }

        // Create segments
        final List<LeafReaderContext> leafReaderContexts = new ArrayList<>();
        final int numDocsInSegment = 1000;
        for (int i = 0, j = 0, docBase = 0; i < numSegments; ++i, ++j, docBase += numDocsInSegment) {
            // Make mock for leaf reader context
            final SegmentReader mockSegmentReader = mock(SegmentReader.class);
            when(mockSegmentReader.getSegmentName()).thenReturn("_" + i + "_165_target_field.faiss");
            when(mockSegmentReader.maxDoc()).thenReturn(numDocsInSegment);

            final LeafReaderContext leafReaderContext = createLeafReaderContext(i, docBase, mockSegmentReader);
            when(mockSegmentReader.getContext()).thenReturn(leafReaderContext);

            leafReaderContexts.add(leafReaderContext);

            // Return answer set per this segment
            when(knnWeight.searchLeaf(eq(leafReaderContext), anyInt())).thenReturn(perLeafResults.get(i));
            when(knnWeight.approximateSearch(eq(leafReaderContext), any(), anyInt(), anyInt())).thenReturn(
                perLeafResults.get(i).getResult()
            );
        }

        when(parentIndexReader.leaves()).thenReturn(leafReaderContexts);

        // Create a weight and do search
        final Weight weight = query.createWeight(searcher, ScoreMode.TOP_DOCS_WITH_SCORES, 1.0f);

        // Validate reentering
        for (int i = 0; i < leafReaderContexts.size(); ++i) {
            // Make mock for leaf reader context
            final LeafReaderContext mockLeafReaderContext = leafReaderContexts.get(i);

            verify(knnWeight, times(1)).searchLeaf(eq(mockLeafReaderContext), anyInt());

            if (i < numSegmentsForReentering) {
                // Even a segment has potential, if the results gotten from exact search, then we must not reenter
                final int expectedInvocations = isApproximateSearch ? 1 : 0;

                // For competitive segments, it should be revisited.
                verify(knnWeight, times(expectedInvocations)).approximateSearch(eq(mockLeafReaderContext), any(), anyInt(), anyInt());
            }
        }

        // Validate results
        // Take top-k for answer set
        List<Float> answerScores = new ArrayList<>();
        for (List<ScoreDoc> scoreDocs : searchResults) {
            for (ScoreDoc scoreDoc : scoreDocs) {
                answerScores.add(scoreDoc.score);
            }
        }
        answerScores.sort((a, b) -> Float.compare(b, a));

        // Collect scores and sort them in desc.
        final List<Float> acquiredScores = new ArrayList<>();
        for (final LeafReaderContext leafReaderContext : leafReaderContexts) {
            final Scorer scorer = weight.scorer(leafReaderContext);
            final DocIdSetIterator iterator = scorer.iterator();
            while (iterator.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                final float score = scorer.score();
                acquiredScores.add(score);
            }
        }
        acquiredScores.sort((a, b) -> Float.compare(b, a));

        // Scores should be the same
        assertEquals("Invalid scores acquired. Answer=" + answerScores + ", got=" + acquiredScores, answerScores, acquiredScores);
    }

    private static LeafReaderContext createLeafReaderContext(final int ord, final int docBase, SegmentReader mockSegmentReader) {
        try {
            // Get the package-private constructor
            Constructor<LeafReaderContext> ctor = LeafReaderContext.class.getDeclaredConstructor(
                CompositeReaderContext.class,
                LeafReader.class,
                int.class,
                int.class,
                int.class,
                int.class
            );
            ctor.setAccessible(true);

            // Call constructor with desired values
            return ctor.newInstance(null, mockSegmentReader, ord, docBase, ord, docBase);
        } catch (Exception e) {
            throw new RuntimeException("Failed to create LeafReaderContext via reflection", e);
        }
    }
}
