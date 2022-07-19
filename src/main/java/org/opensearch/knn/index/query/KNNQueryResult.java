/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

/**
 * Place holder for the score of the document
 */
public class KNNQueryResult {
    private final int id;
    private final float score;

    public KNNQueryResult(final int id, final float score) {
        this.id = id;
        this.score = score;
    }

    public int getId() {
        return this.id;
    }

    public float getScore() {
        return this.score;
    }
}
