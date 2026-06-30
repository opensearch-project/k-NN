/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.junit.Test;
import org.opensearch.knn.index.query.memoryoptsearch.RadiusVectorSimilarityCollector;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

public class RadiusVectorSimilarityCollectorTests {

    private static final float TRAVERSAL_SIMILARITY = 0.5f;
    private static final float RESULT_SIMILARITY = 0.8f;
    private static final long VISIT_LIMIT = 1000L;

    @Test
    public void testDefaultConstructor_usesHnswStrategy() {
        final RadiusVectorSimilarityCollector collector = new RadiusVectorSimilarityCollector(
            TRAVERSAL_SIMILARITY,
            RESULT_SIMILARITY,
            VISIT_LIMIT
        );
        assertTrue(
            "Expected default Hnsw search strategy but got " + collector.getSearchStrategy(),
            collector.getSearchStrategy() instanceof KnnSearchStrategy.Hnsw
        );
    }

    @Test
    public void testStrategyConstructor_forwardsSeededStrategyToCollector() {
        // A Seeded strategy is what enables re-entrant radial search (entry points from a prior phase).
        final DocIdSetIterator seedDocs = DocIdSetIterator.all(5);
        final KnnSearchStrategy.Seeded seededStrategy = new KnnSearchStrategy.Seeded(seedDocs, 5, KnnSearchStrategy.Hnsw.DEFAULT);

        final RadiusVectorSimilarityCollector collector = new RadiusVectorSimilarityCollector(
            TRAVERSAL_SIMILARITY,
            RESULT_SIMILARITY,
            VISIT_LIMIT,
            seededStrategy
        );

        assertSame(
            "RadiusVectorSimilarityCollector must forward the provided search strategy so seeding takes effect",
            seededStrategy,
            collector.getSearchStrategy()
        );
    }

    @Test
    public void testCollect_onlyCollectsDocsAtOrAboveResultSimilarity() {
        final RadiusVectorSimilarityCollector collector = new RadiusVectorSimilarityCollector(
            TRAVERSAL_SIMILARITY,
            RESULT_SIMILARITY,
            VISIT_LIMIT,
            KnnSearchStrategy.Hnsw.DEFAULT
        );

        // Below result similarity -> dropped
        collector.collect(1, 0.6f);
        // At result similarity -> collected
        collector.collect(2, RESULT_SIMILARITY);
        // Above result similarity -> collected
        collector.collect(3, 0.95f);

        final TopDocs topDocs = collector.topDocs();
        assertEquals(2, topDocs.scoreDocs.length);
        for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
            assertTrue("Collected docs must have similarity >= resultSimilarity", scoreDoc.score >= RESULT_SIMILARITY);
        }
    }

    @Test
    public void testConstructor_rejectsTraversalGreaterThanResult() {
        assertThrows(
            IllegalArgumentException.class,
            () -> new RadiusVectorSimilarityCollector(0.9f, 0.5f, VISIT_LIMIT, KnnSearchStrategy.Hnsw.DEFAULT)
        );
        assertThrows(IllegalArgumentException.class, () -> new RadiusVectorSimilarityCollector(0.9f, 0.5f, VISIT_LIMIT));
    }
}
