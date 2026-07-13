/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.memoryoptsearch;

import org.apache.lucene.search.AbstractKnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.knn.KnnSearchStrategy;

import java.util.ArrayList;
import java.util.List;

/**
 * Collector for radial (similarity-threshold) search over an HNSW graph.
 * <p>
 * Uses a fixed similarity threshold as a hard cutoff: collects all visited nodes at or above
 * the threshold and refuses to explore below it. This is designed for the two-phase radial
 * search where phase-1 seeds provide good entry points, so no exploration beyond the threshold
 * is needed.
 * <p>
 * Compatible with Lucene 10.5's {@code HnswGraphSearcher} which computes the traversal bound as
 * {@code Math.nextUp(minCompetitiveSimilarity())}. We return {@code Math.nextDown(similarity)}
 * so the effective cutoff equals the threshold exactly.
 */
public class RadiusVectorSimilarityCollector extends AbstractKnnCollector {
    private static final KnnSearchStrategy.Hnsw DEFAULT_STRATEGY = new KnnSearchStrategy.Hnsw(0);

    private final float similarity;
    private final List<ScoreDoc> scoreDocList;

    /**
     * @param similarity minimum similarity for both traversal and collection.
     * @param visitLimit limit on number of nodes to visit.
     */
    public RadiusVectorSimilarityCollector(float similarity, long visitLimit) {
        this(similarity, visitLimit, DEFAULT_STRATEGY);
    }

    /**
     * @param similarity     minimum similarity for both traversal and collection.
     * @param visitLimit     limit on number of nodes to visit.
     * @param searchStrategy the HNSW search strategy (e.g. {@link KnnSearchStrategy.Seeded}).
     */
    public RadiusVectorSimilarityCollector(float similarity, long visitLimit, KnnSearchStrategy searchStrategy) {
        super(1, visitLimit, searchStrategy);
        this.similarity = similarity;
        this.scoreDocList = new ArrayList<>();
    }

    @Override
    public boolean collect(int docId, float similarity) {
        if (similarity >= this.similarity) {
            scoreDocList.add(new ScoreDoc(docId, similarity));
        }
        // Return false: our minCompetitiveSimilarity is constant, no re-read needed.
        return false;
    }

    @Override
    public float minCompetitiveSimilarity() {
        // Lucene 10.5's HnswGraphSearcher uses Math.nextUp(this) as the traversal bound.
        // Returning nextDown(similarity) makes the effective cutoff exactly equal to similarity.
        return Math.nextDown(similarity);
    }

    @Override
    public TopDocs topDocs() {
        TotalHits.Relation relation = earlyTerminated() ? TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO : TotalHits.Relation.EQUAL_TO;
        return new TopDocs(new TotalHits(visitedCount(), relation), scoreDocList.toArray(ScoreDoc[]::new));
    }

    @Override
    public int numCollected() {
        return scoreDocList.size();
    }
}
