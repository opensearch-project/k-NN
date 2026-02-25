/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.search;

import lombok.EqualsAndHashCode;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.search.deciders.ConcurrentSearchDecision;
import org.opensearch.search.deciders.ConcurrentSearchRequestDecider;

import java.util.Optional;

/**
 * Decides if the knn query uses concurrent segment search
 * As of 2.17, this is only used when
 *  - "index.search.concurrent_segment_search.mode": "auto" or
 *  - "search.concurrent_segment_search.mode": "auto"
 *
 *  Note: the class is not thread-safe and a new instance needs to be created for each request
 */
@EqualsAndHashCode(callSuper = true)
public class KNNConcurrentSearchRequestDecider extends ConcurrentSearchRequestDecider {

    private static final ConcurrentSearchDecision DEFAULT_KNN_DECISION = new ConcurrentSearchDecision(
        ConcurrentSearchDecision.DecisionStatus.NO_OP,
        "Default decision"
    );
    private static final ConcurrentSearchDecision YES = new ConcurrentSearchDecision(
        ConcurrentSearchDecision.DecisionStatus.YES,
        "Enable concurrent search for knn as Query has k-NN query in it and index is k-nn index"
    );

    private ConcurrentSearchDecision knnDecision = DEFAULT_KNN_DECISION;

    @Override
    public void evaluateForQuery(final QueryBuilder queryBuilder, final IndexSettings indexSettings) {
        if (queryBuilder instanceof KNNQueryBuilder && indexSettings.getValue(KNNSettings.IS_KNN_INDEX_SETTING)) {
            knnDecision = YES;
        } else {
            knnDecision = DEFAULT_KNN_DECISION;
        }
    }

    @Override
    public ConcurrentSearchDecision getConcurrentSearchDecision() {
        return knnDecision;
    }

    /**
     * Returns {@link KNNConcurrentSearchRequestDecider} when index.knn is true
     */
    public static class Factory implements ConcurrentSearchRequestDecider.Factory {
        public Optional<ConcurrentSearchRequestDecider> create(final IndexSettings indexSettings) {
            if (indexSettings.getValue(KNNSettings.IS_KNN_INDEX_SETTING)) {
                return Optional.of(new KNNConcurrentSearchRequestDecider());
            }
            return Optional.empty();
        }
    }
}
