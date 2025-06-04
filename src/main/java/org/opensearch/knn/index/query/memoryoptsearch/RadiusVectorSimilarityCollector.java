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
 * Clone of Lucene's VectorSimilarityCollector, which cannot be used directly due to its package-private visibility.
 */
public class RadiusVectorSimilarityCollector extends AbstractKnnCollector {
    private static final KnnSearchStrategy.Hnsw DEFAULT_STRATEGY = new KnnSearchStrategy.Hnsw(0);

    private final float traversalSimilarity, resultSimilarity;
    private float maxSimilarity;
    private final List<ScoreDoc> scoreDocList;

    /**
     * Perform a similarity-based graph search. The graph is traversed till better scoring nodes are
     * available, or the best candidate is below {@link #traversalSimilarity}. All traversed nodes
     * above {@link #resultSimilarity} are collected.
     *
     * @param traversalSimilarity (lower) similarity score for graph traversal.
     * @param resultSimilarity (higher) similarity score for result collection.
     * @param visitLimit limit on number of nodes to visit.
     */
    public RadiusVectorSimilarityCollector(float traversalSimilarity, float resultSimilarity, long visitLimit) {
        // TODO: add search strategy support
        super(1, visitLimit, DEFAULT_STRATEGY);
        if (traversalSimilarity > resultSimilarity) {
            throw new IllegalArgumentException("traversalSimilarity should be <= resultSimilarity");
        }
        this.traversalSimilarity = traversalSimilarity;
        this.resultSimilarity = resultSimilarity;
        this.maxSimilarity = Float.NEGATIVE_INFINITY;
        this.scoreDocList = new ArrayList<>();
    }

    @Override
    public boolean collect(int docId, float similarity) {
        maxSimilarity = Math.max(maxSimilarity, similarity);
        if (similarity >= resultSimilarity) {
            scoreDocList.add(new ScoreDoc(docId, similarity));
        }
        return true;
    }

    @Override
    public float minCompetitiveSimilarity() {
        return Math.min(traversalSimilarity, maxSimilarity);
    }

    @Override
    public TopDocs topDocs() {
        // Results are not returned in a sorted order to prevent unnecessary calculations (because we do
        // not need to maintain the topK)
        TotalHits.Relation relation = earlyTerminated() ? TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO : TotalHits.Relation.EQUAL_TO;
        return new TopDocs(new TotalHits(visitedCount(), relation), scoreDocList.toArray(ScoreDoc[]::new));
    }

    @Override
    public int numCollected() {
        return scoreDocList.size();
    }
}
