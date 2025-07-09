/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profile.query;

import java.util.Locale;

public enum KNNQueryTimingType {
    ANN_SEARCH,
    EXACT_SEARCH,
    BITSET_CREATION,
    EXACT_SEARCH_AFTER_ANN,
    EXACT_SEARCH_AFTER_FILTER;

    @Override
    public String toString() {
        return name().toLowerCase(Locale.ROOT);
    }
}
