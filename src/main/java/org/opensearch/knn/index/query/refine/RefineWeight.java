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

import lombok.Getter;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.FilterWeight;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.Weight;

import java.io.IOException;

/**
 * Weight implementation for refining results of a query with higher precision computations.
 *TODO: Not sure that the delegate method is the right call here. Need to check and probably implement
 * other methods.
 */
@Getter
public class RefineWeight extends FilterWeight {
    private final RefineContext refineContext;
    private final float boost;

    /**
     *
     * @param delegateWeight wrapped query weight function
     * @param refineContext context for re-scoring
     */
    protected RefineWeight(Weight delegateWeight, RefineContext refineContext, float boost) {
        super(delegateWeight);
        this.refineContext = refineContext;
        this.boost = boost;
    }

    @Override
    public Scorer scorer(LeafReaderContext leafReaderContext) throws IOException {
        Scorer subQueryScorer = in.scorer(leafReaderContext);
        return new RefineScorer(subQueryScorer, refineContext, refineContext.getKNNVectorValues(leafReaderContext), boost);
    }
}
