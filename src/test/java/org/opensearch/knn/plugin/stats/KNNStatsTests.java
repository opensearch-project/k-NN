/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats;

import org.opensearch.knn.KNNTestCase;

import java.util.Map;

/**
 * Unit tests for {@link KNNStats}, focusing on the remote vector index build stats map which surfaces the
 * {@link KNNRemoteIndexBuildValue} counters (including the merge-abort and terminal-exception counters) through
 * the {@code _stats} API.
 */
public class KNNStatsTests extends KNNTestCase {

    @Override
    public void tearDown() throws Exception {
        super.tearDown();
        for (KNNRemoteIndexBuildValue value : KNNRemoteIndexBuildValue.values()) {
            value.reset();
        }
    }

    /**
     * The remote vector index build stats expose client, repository, and build sub-maps, and the client sub-map
     * surfaces every client-side counter including the benign-termination counters added for early merge abort.
     */
    @SuppressWarnings("unchecked")
    public void testRemoteIndexBuildStatsMapContainsAllCounters() {
        KNNStats knnStats = new KNNStats();

        Object remoteStatsValue = knnStats.getStats().get(StatNames.REMOTE_VECTOR_INDEX_BUILD_STATS.getName()).getValue();
        assertTrue(remoteStatsValue instanceof Map);
        Map<String, Map<String, Object>> remoteStats = (Map<String, Map<String, Object>>) remoteStatsValue;

        // The three sub-maps must all be present.
        assertTrue(remoteStats.containsKey(StatNames.CLIENT_STATS.getName()));
        assertTrue(remoteStats.containsKey(StatNames.REPOSITORY_STATS.getName()));
        assertTrue(remoteStats.containsKey(StatNames.BUILD_STATS.getName()));

        Map<String, Object> clientStats = remoteStats.get(StatNames.CLIENT_STATS.getName());
        assertTrue(clientStats.containsKey(KNNRemoteIndexBuildValue.INDEX_BUILD_SUCCESS_COUNT.getName()));
        assertTrue(clientStats.containsKey(KNNRemoteIndexBuildValue.INDEX_BUILD_FAILURE_COUNT.getName()));
        // Benign-termination counters added for the early merge-abort feature.
        assertTrue(clientStats.containsKey(KNNRemoteIndexBuildValue.INDEX_BUILD_MERGE_ABORT_EXCEPTION.getName()));
        assertTrue(clientStats.containsKey(KNNRemoteIndexBuildValue.INDEX_BUILD_TERMINAL_EXCEPTION.getName()));
    }

    /**
     * The stats map is backed by a supplier, so incrementing the underlying counters is reflected the next time the
     * stat value is read.
     */
    @SuppressWarnings("unchecked")
    public void testRemoteIndexBuildStatsReflectCounterValues() {
        KNNRemoteIndexBuildValue.INDEX_BUILD_MERGE_ABORT_EXCEPTION.increment();
        KNNRemoteIndexBuildValue.INDEX_BUILD_TERMINAL_EXCEPTION.increment();
        KNNRemoteIndexBuildValue.INDEX_BUILD_TERMINAL_EXCEPTION.increment();

        KNNStats knnStats = new KNNStats();
        Map<String, Map<String, Object>> remoteStats = (Map<String, Map<String, Object>>) knnStats.getStats()
            .get(StatNames.REMOTE_VECTOR_INDEX_BUILD_STATS.getName())
            .getValue();
        Map<String, Object> clientStats = remoteStats.get(StatNames.CLIENT_STATS.getName());

        assertEquals(1L, (long) (Long) clientStats.get(KNNRemoteIndexBuildValue.INDEX_BUILD_MERGE_ABORT_EXCEPTION.getName()));
        assertEquals(2L, (long) (Long) clientStats.get(KNNRemoteIndexBuildValue.INDEX_BUILD_TERMINAL_EXCEPTION.getName()));
        assertEquals(0L, (long) (Long) clientStats.get(KNNRemoteIndexBuildValue.INDEX_BUILD_FAILURE_COUNT.getName()));
    }
}
