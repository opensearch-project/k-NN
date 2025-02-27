/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats;

import java.util.HashMap;
import java.util.Map;

/**
 * Additional context needed for fetching KNN stats.
 */
public class KNNStatFetchContext {
    private final Map<String, Map<String, Object>> contexts;

    public KNNStatFetchContext() {
        this.contexts = new HashMap<>();
    }

    public void addContext(String statName, Map<String, Object> context) {
        this.contexts.put(statName, context);
    }

    public Map<String, Object> getContext(String statName) {
        return this.contexts.get(statName);
    }
}
