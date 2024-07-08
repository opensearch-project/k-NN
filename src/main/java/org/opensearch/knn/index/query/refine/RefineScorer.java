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

package org.opensearch.knn.index.query.refine;

import org.apache.lucene.search.FilterScorer;
import org.apache.lucene.search.Scorer;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.io.IOException;

/**
 * Calculates the refinement score on top of existing scorer.
 */
public class RefineScorer extends FilterScorer {
    private final RefineContext refineContext;
    private final KNNVectorValues<float[]> knnVectorValues;
    private final float boost;

    /**
     *
     * @param delegateScorer scorer for wrapped Query
     * @param refineContext context for refinement logic
     * @param knnVectorValues vector values for KNN vector to do refining
     * @param boost factor to multiply score by
     */
    public RefineScorer(Scorer delegateScorer, RefineContext refineContext, KNNVectorValues<float[]> knnVectorValues, float boost) {
        super(delegateScorer);
        this.refineContext = refineContext;
        this.knnVectorValues = knnVectorValues;
        this.boost = boost;
    }

    @Override
    public float score() throws IOException {
        knnVectorValues.getVectorValuesIterator().advance(docID());
        return refineContext.refine(knnVectorValues.getVector()) * boost;
    }

    @Override
    public float getMaxScore(int upTo) {
        return Float.MAX_VALUE;
    }
}
