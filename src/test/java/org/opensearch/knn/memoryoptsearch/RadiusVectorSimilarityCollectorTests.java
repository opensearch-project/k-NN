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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

public class RadiusVectorSimilarityCollectorTests {

    private static final float SIMILARITY = 0.8f;
    private static final long VISIT_LIMIT = 1000L;

    @Test
    public void testDefaultConstructor_usesHnswStrategy() {
        final RadiusVectorSimilarityCollector collector = new RadiusVectorSimilarityCollector(SIMILARITY, VISIT_LIMIT);
        assertTrue(
            "Expected default Hnsw search strategy but got " + collector.getSearchStrategy(),
            collector.getSearchStrategy() instanceof KnnSearchStrategy.Hnsw
        );
    }

    @Test
    public void testStrategyConstructor_forwardsSeededStrategyToCollector() {
        final DocIdSetIterator seedDocs = DocIdSetIterator.all(5);
        final KnnSearchStrategy.Seeded seededStrategy = new KnnSearchStrategy.Seeded(seedDocs, 5, KnnSearchStrategy.Hnsw.DEFAULT);

        final RadiusVectorSimilarityCollector collector = new RadiusVectorSimilarityCollector(
            SIMILARITY,
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
    public void testCollect_onlyCollectsDocsAtOrAboveThreshold() {
        final RadiusVectorSimilarityCollector collector = new RadiusVectorSimilarityCollector(
            SIMILARITY,
            VISIT_LIMIT,
            KnnSearchStrategy.Hnsw.DEFAULT
        );

        // Below threshold -> dropped
        assertFalse(collector.collect(1, 0.6f));
        // At threshold -> collected
        assertFalse(collector.collect(2, SIMILARITY));
        // Above threshold -> collected
        assertFalse(collector.collect(3, 0.95f));

        final TopDocs topDocs = collector.topDocs();
        assertEquals(2, topDocs.scoreDocs.length);
        for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
            assertTrue("Collected docs must have similarity >= threshold", scoreDoc.score >= SIMILARITY);
        }
    }

    @Test
    public void testMinCompetitiveSimilarity_isConstantHardCutoff() {
        final RadiusVectorSimilarityCollector collector = new RadiusVectorSimilarityCollector(
            SIMILARITY,
            VISIT_LIMIT,
            KnnSearchStrategy.Hnsw.DEFAULT
        );

        // Before any collection, the bound is already set as a hard cutoff.
        float expected = Math.nextDown(SIMILARITY);
        assertEquals(expected, collector.minCompetitiveSimilarity(), 0f);

        // After collecting, the bound doesn't change.
        collector.collect(1, 0.95f);
        assertEquals(expected, collector.minCompetitiveSimilarity(), 0f);
    }
}
