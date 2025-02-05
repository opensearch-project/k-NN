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

    @Getter
    private final Map<Object, Integer> annResultPerLeaf;

    @Getter
    private final Map<Integer, Float> rawScores;

    @Getter
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
}
