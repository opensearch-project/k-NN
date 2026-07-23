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
 * Clone of Lucene's {@code VectorSimilarityCollector}, which cannot be used directly due to its
 * package-private visibility.
 *
 * <p>This mirrors Lucene 10.5's decay-based radial search: instead of a fixed lower
 * {@code traversalSimilarity} threshold, the graph-traversal buffer is adaptive. It starts high and
 * decays towards the scores of nodes that were traversed but not collected, using the provided
 * {@code decay} factor. The decay factor lies in {@code [0, 1]}; higher values explore more of the
 * graph for better recall. All traversed nodes at or above {@code resultSimilarity} are collected.
 */
public class RadiusVectorSimilarityCollector extends AbstractKnnCollector {
    private static final KnnSearchStrategy.Hnsw DEFAULT_STRATEGY = new KnnSearchStrategy.Hnsw(0);

    // Bounds for the decay factor, matching Lucene's AbstractVectorSimilarityQuery.
    static final float DECAY_MAX_APPROXIMATION = 0f;
    static final float DECAY_MAX_QUALITY = 1f;

    private final float resultSimilarity, decay;
    private final List<ScoreDoc> scoreDocList;
    private float minCompetitiveSimilarity;

    /**
     * Perform a decay-based, similarity-based graph search. The graph is traversed while candidates
     * remain competitive with the adaptive buffer; all traversed nodes at or above
     * {@link #resultSimilarity} are collected.
     *
     * @param resultSimilarity similarity score for result collection.
     * @param decay decay factor for the graph-traversal buffer, in range {@code [0, 1]}.
     * @param visitLimit limit on number of nodes to visit.
     */
    public RadiusVectorSimilarityCollector(float resultSimilarity, float decay, long visitLimit) {
        // TODO: add search strategy support
        super(1, visitLimit, DEFAULT_STRATEGY);
        if (Float.isNaN(resultSimilarity)) {
            throw new IllegalArgumentException("resultSimilarity must have a valid value; got " + resultSimilarity);
        }
        if (Float.isNaN(decay) || decay < DECAY_MAX_APPROXIMATION || decay > DECAY_MAX_QUALITY) {
            throw new IllegalArgumentException(
                "decay must lie in range [DECAY_MAX_APPROXIMATION = 0, DECAY_MAX_QUALITY = 1]; got " + decay
            );
        }
        this.resultSimilarity = resultSimilarity;
        this.decay = decay;
        this.scoreDocList = new ArrayList<>();
        this.minCompetitiveSimilarity = Math.nextUp(Float.NEGATIVE_INFINITY);
    }

    @Override
    public boolean collect(int docId, float similarity) {
        if (similarity >= resultSimilarity) {
            scoreDocList.add(new ScoreDoc(docId, similarity));
        } else if (decay < DECAY_MAX_QUALITY) {
            // Decay the traversal buffer towards the score of the current (uncollected) node.
            minCompetitiveSimilarity = (float) (similarity + ((double) minCompetitiveSimilarity - similarity) * decay);
            return true;
        }
        return false;
    }

    @Override
    public float minCompetitiveSimilarity() {
        return minCompetitiveSimilarity;
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
