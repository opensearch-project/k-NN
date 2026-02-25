/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.Getter;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.plugin.stats.KNNCounter;

@Getter
public enum VectorQueryType {
    K(KNNConstants.K) {
        @Override
        public KNNCounter getQueryStatCounter() {
            return KNNCounter.KNN_QUERY_REQUESTS;
        }

        @Override
        public KNNCounter getQueryWithFilterStatCounter() {
            return KNNCounter.KNN_QUERY_WITH_FILTER_REQUESTS;
        }
    },
    MIN_SCORE(KNNConstants.MIN_SCORE) {
        @Override
        public KNNCounter getQueryStatCounter() {
            return KNNCounter.MIN_SCORE_QUERY_REQUESTS;
        }

        @Override
        public KNNCounter getQueryWithFilterStatCounter() {
            return KNNCounter.MIN_SCORE_QUERY_WITH_FILTER_REQUESTS;
        }
    },
    MAX_DISTANCE(KNNConstants.MAX_DISTANCE) {
        @Override
        public KNNCounter getQueryStatCounter() {
            return KNNCounter.MAX_DISTANCE_QUERY_REQUESTS;
        }

        @Override
        public KNNCounter getQueryWithFilterStatCounter() {
            return KNNCounter.MAX_DISTANCE_QUERY_WITH_FILTER_REQUESTS;
        }
    };

    private final String queryTypeName;

    VectorQueryType(String queryTypeName) {
        this.queryTypeName = queryTypeName;
    }

    public abstract KNNCounter getQueryStatCounter();

    public abstract KNNCounter getQueryWithFilterStatCounter();

    public boolean isRadialSearch() {
        return this == MAX_DISTANCE || this == MIN_SCORE;
    }
}
