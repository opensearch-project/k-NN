/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profile.query;

import java.util.Locale;

public enum LuceneEngineKnnTimingType {
    EXACT_SEARCH,
    ANN_SEARCH,
    EXPAND_NESTED_ANN,
    BITSET_CREATION,
    EXPAND_NESTED_EXACT,
    INTERNAL_EXACT,
    RESCORE;

    @Override
    public String toString() {
        return name().toLowerCase(Locale.ROOT);
    }
}
