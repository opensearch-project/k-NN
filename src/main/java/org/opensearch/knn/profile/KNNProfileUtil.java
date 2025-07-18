/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profile;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.util.IOSupplier;
import org.opensearch.search.internal.ContextIndexSearcher;
import org.opensearch.search.profile.ContextualProfileBreakdown;
import org.opensearch.search.profile.Timer;
import org.opensearch.search.profile.query.QueryProfiler;

import java.io.IOException;

/**
 * Utility class for profiling KNN plugin.
 */
public class KNNProfileUtil {
    /**
     * Gets the query profiler from an index searcher
     * @param searcher
     * @return {@link org.opensearch.search.profile.query.QueryProfiler}
     */
    public static QueryProfiler getProfiler(IndexSearcher searcher) {
        if (searcher instanceof ContextIndexSearcher contextIndexSearcher) {
            if (contextIndexSearcher.getProfiler() != null) {
                return contextIndexSearcher.getProfiler();
            }
        }
        return null;
    }

    /**
     * Executes the action provided by the supplier and times it based on the provided timing type.
     * @param profile
     * @param leafReaderContext
     * @param timingType
     * @param action
     * @return result of the supplier
     */
    public static Object profile(
        ContextualProfileBreakdown profile,
        LeafReaderContext leafReaderContext,
        Enum<?> timingType,
        IOSupplier<?> action
    ) throws IOException {
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

    /**
     * Executes the action provided by the supplier and times it based on the provided timing type.
     * @param profiler
     * @param leafReaderContext
     * @param timingType
     * @param action
     * @return result of the supplier
     */
    public static Object profile(
        QueryProfiler profiler,
        Query query,
        LeafReaderContext leafReaderContext,
        Enum<?> timingType,
        IOSupplier<?> action
    ) throws IOException {
        if (profiler != null) {
            ContextualProfileBreakdown profile = (ContextualProfileBreakdown) profiler.getProfileBreakdown(query);
            return profile(profile, leafReaderContext, timingType, action);
        }
        return action.get();
    }
}
