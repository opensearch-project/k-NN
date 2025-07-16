/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profile;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.opensearch.search.internal.ContextIndexSearcher;
import org.opensearch.search.profile.ContextualProfileBreakdown;
import org.opensearch.search.profile.Timer;
import org.opensearch.search.profile.query.QueryProfiler;

import java.util.function.Supplier;

public class KNNProfileUtil {
    public static QueryProfiler getProfiler(IndexSearcher searcher) {
        if (searcher instanceof ContextIndexSearcher contextIndexSearcher) {
            if (contextIndexSearcher.getProfiler() != null) {
                return contextIndexSearcher.getProfiler();
            }
        }
        return null;
    }

    public static Object profile(
        ContextualProfileBreakdown profile,
        LeafReaderContext leafReaderContext,
        Enum<?> timingType,
        Supplier<?> action
    ) {
        if (profile != null) {
            Timer timer = profile.context(leafReaderContext).getTimer(timingType);
            timer.start();
            try {
                return action.get();
            } finally {
                timer.stop();
            }
        }
        return action.get();
    }

    public static Object profile(
        QueryProfiler profiler,
        Query query,
        LeafReaderContext leafReaderContext,
        Enum<?> timingType,
        Supplier<?> action
    ) {
        if (profiler != null) {
            ContextualProfileBreakdown profile = (ContextualProfileBreakdown) profiler.getProfileBreakdown(query);
            return profile(profile, leafReaderContext, timingType, action);
        }
        return action.get();
    }
}
