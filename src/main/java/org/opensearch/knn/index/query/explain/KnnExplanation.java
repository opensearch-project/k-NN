/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.explain;

import lombok.Getter;
import lombok.Setter;
import org.opensearch.knn.index.query.KNNScorer;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * This class captures details around knn explain queries that is used
 * by explain API to generate explanation for knn queries
 */
public class KnnExplanation {

    private final Map<Object, Integer> annResultPerLeaf;

    private final Map<Integer, Float> rawScores;

    private final Map<Object, KNNScorer> knnScorerPerLeaf;

    @Setter
    @Getter
    private int cardinality;

    public KnnExplanation() {
        this.annResultPerLeaf = new ConcurrentHashMap<>();
        this.rawScores = new ConcurrentHashMap<>();
        this.knnScorerPerLeaf = new ConcurrentHashMap<>();
        this.cardinality = 0;
    }

    public void addLeafResult(Object leafId, int annResult) {
        this.annResultPerLeaf.put(leafId, annResult);
    }

    public void addRawScore(int docId, float rawScore) {
        this.rawScores.put(docId, rawScore);
    }

    public void addKnnScorer(Object leafId, KNNScorer knnScorer) {
        this.knnScorerPerLeaf.put(leafId, knnScorer);
    }

    public Integer getAnnResult(Object leafId) {
        return this.annResultPerLeaf.get(leafId);
    }

    public Float getRawScore(int docId) {
        return this.rawScores.get(docId);
    }

    public KNNScorer getKnnScorer(Object leafId) {
        return this.knnScorerPerLeaf.get(leafId);
    }
}
