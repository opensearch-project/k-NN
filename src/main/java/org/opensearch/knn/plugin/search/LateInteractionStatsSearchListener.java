/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.search;

import org.opensearch.index.shard.SearchOperationListener;
import org.opensearch.knn.plugin.stats.KNNCounter;
import org.opensearch.search.builder.SearchSourceBuilder;
import org.opensearch.search.internal.SearchContext;

/**
 * Tracks domain adoption of the late interaction scoring function.
 *
 * <p>The {@code lateInteractionScore} Painless function is a static, per-document
 * whitelisted method, so incrementing inside it counts once per scored document
 * (a single request scoring N docs would count N times). To measure adoption at
 * the request level, this listener instead increments the counter once per
 * shard-level query phase whose source uses the function. This matches the
 * granularity of the other k-NN query counters (e.g. {@code knn_query_requests}),
 * which are also node/shard-level, and it does not touch the Painless API.
 */
public class LateInteractionStatsSearchListener implements SearchOperationListener {

    private static final String LATE_INTERACTION_MARKER = "lateInteractionScore";

    @Override
    public void onPreQueryPhase(SearchContext searchContext) {
        if (searchContext == null || searchContext.request() == null) {
            return;
        }
        SearchSourceBuilder source = searchContext.request().source();
        if (source == null) {
            return;
        }
        // Count once per shard query that uses the late interaction painless function.
        if (source.toString().contains(LATE_INTERACTION_MARKER)) {
            KNNCounter.LATE_INTERACTION_SCORE_REQUESTS.increment();
        }
    }
}
