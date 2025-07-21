/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profile.query;

import java.util.Locale;

/**
 * Timers used for profiling KNN queries
 */
public enum KNNQueryTimingType {
    ANN_SEARCH,
    EXACT_SEARCH,
    GRAPH_LOAD,
    BITSET_CREATION;

    @Override
    public String toString() {
        return name().toLowerCase(Locale.ROOT);
    }
}
