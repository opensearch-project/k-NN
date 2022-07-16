/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
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
