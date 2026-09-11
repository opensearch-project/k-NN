/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;

/**
 * Per-index, cumulative counters of remote vector index build outcomes (success / failure), maintained on
 * the data node in the codec flush/merge thread.
 *
 * <p>Unlike a per-segment {@code SegmentInfo} attribute, these counters SURVIVE Lucene segment merges: a
 * remote build that failed (and fell back to local) on an intermediate segment is counted permanently,
 * even if that segment is subsequently merged away. This is what allows a test to deterministically assert
 * that EVERY remote build attempt for an index succeeded (successCount &gt; 0 and failureCount == 0),
 * closing the "merged-away failure is invisible in GET &lt;index&gt;/_segments" gap.
 *
 * <p>Counters are node-local process state and reset on node restart, which is acceptable for the
 * single-node integration cluster that consumes them. Keyed by OpenSearch index name.
 */
public final class RemoteIndexBuildPerIndexStats {

    private static final Map<String, LongAdder> SUCCESS_BY_INDEX = new ConcurrentHashMap<>();
    private static final Map<String, LongAdder> FAILURE_BY_INDEX = new ConcurrentHashMap<>();

    private RemoteIndexBuildPerIndexStats() {}

    /** Record a successful remote index build for the given index. */
    public static void incrementSuccess(final String indexName) {
        if (indexName == null) {
            return;
        }
        SUCCESS_BY_INDEX.computeIfAbsent(indexName, k -> new LongAdder()).increment();
    }

    /** Record a failed remote index build (which fell back to a local build) for the given index. */
    public static void incrementFailure(final String indexName) {
        if (indexName == null) {
            return;
        }
        FAILURE_BY_INDEX.computeIfAbsent(indexName, k -> new LongAdder()).increment();
    }

    public static long getSuccessCount(final String indexName) {
        if (indexName == null) {
            return 0L;
        }
        LongAdder a = SUCCESS_BY_INDEX.get(indexName);
        return a == null ? 0L : a.sum();
    }

    public static long getFailureCount(final String indexName) {
        if (indexName == null) {
            return 0L;
        }
        LongAdder a = FAILURE_BY_INDEX.get(indexName);
        return a == null ? 0L : a.sum();
    }

    /**
     * Snapshot of per-index counts for stats exposure, shaped as
     * {@code { <indexName>: { index_build_success_count: N, index_build_failure_count: M }, ... }}.
     */
    public static Map<String, Object> asMap() {
        Map<String, Object> out = new LinkedHashMap<>();
        for (String index : SUCCESS_BY_INDEX.keySet()) {
            out.computeIfAbsent(index, k -> new LinkedHashMap<String, Object>());
        }
        for (String index : FAILURE_BY_INDEX.keySet()) {
            out.computeIfAbsent(index, k -> new LinkedHashMap<String, Object>());
        }
        for (Map.Entry<String, Object> e : out.entrySet()) {
            @SuppressWarnings("unchecked")
            Map<String, Object> perIndex = (Map<String, Object>) e.getValue();
            perIndex.put(KNNRemoteIndexBuildValue.INDEX_BUILD_SUCCESS_COUNT.getName(), getSuccessCount(e.getKey()));
            perIndex.put(KNNRemoteIndexBuildValue.INDEX_BUILD_FAILURE_COUNT.getName(), getFailureCount(e.getKey()));
        }
        return Collections.unmodifiableMap(out);
    }
}
