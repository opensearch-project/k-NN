/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats;

import org.opensearch.knn.KNNTestCase;

import java.util.Map;
import java.util.UUID;

/**
 * Unit tests for {@link RemoteIndexBuildPerIndexStats}, the per-index, merge-surviving remote-build
 * success/failure counters. These lock in the counting contract that the integration assertion
 * (KNNRestTestCase#verifyRemoteIndexBuild) relies on to catch failed builds on segments that are later
 * merged away.
 */
public class RemoteIndexBuildPerIndexStatsTests extends KNNTestCase {

    // Use a unique index name per test so the process-global counters do not collide with other tests.
    private String uniqueIndex() {
        return "idx-" + UUID.randomUUID();
    }

    public void testUnknownIndexReturnsZero() {
        String index = uniqueIndex();
        assertEquals(0L, RemoteIndexBuildPerIndexStats.getSuccessCount(index));
        assertEquals(0L, RemoteIndexBuildPerIndexStats.getFailureCount(index));
    }

    public void testIncrementSuccessCountsPerIndex() {
        String index = uniqueIndex();
        for (int i = 1; i <= 5; i++) {
            RemoteIndexBuildPerIndexStats.incrementSuccess(index);
            assertEquals((long) i, RemoteIndexBuildPerIndexStats.getSuccessCount(index));
        }
        // Failures must remain independent of successes.
        assertEquals(0L, RemoteIndexBuildPerIndexStats.getFailureCount(index));
    }

    public void testIncrementFailureCountsPerIndex() {
        String index = uniqueIndex();
        for (int i = 1; i <= 3; i++) {
            RemoteIndexBuildPerIndexStats.incrementFailure(index);
            assertEquals((long) i, RemoteIndexBuildPerIndexStats.getFailureCount(index));
        }
        assertEquals(0L, RemoteIndexBuildPerIndexStats.getSuccessCount(index));
    }

    /**
     * The core guarantee: a mix of successes and failures on the SAME index (as happens when some flush /
     * merge builds succeed remotely and others fall back to local) is fully accounted for. This is exactly
     * the "intermediate segment failed then merged away" case that a surviving-segment scan would miss.
     */
    public void testMixedSuccessAndFailureAreBothCounted() {
        String index = uniqueIndex();
        RemoteIndexBuildPerIndexStats.incrementSuccess(index);
        RemoteIndexBuildPerIndexStats.incrementFailure(index); // this failure must never be lost
        RemoteIndexBuildPerIndexStats.incrementSuccess(index);

        assertEquals(2L, RemoteIndexBuildPerIndexStats.getSuccessCount(index));
        assertEquals(1L, RemoteIndexBuildPerIndexStats.getFailureCount(index));
    }

    public void testCountsAreIsolatedBetweenIndices() {
        String indexA = uniqueIndex();
        String indexB = uniqueIndex();

        RemoteIndexBuildPerIndexStats.incrementSuccess(indexA);
        RemoteIndexBuildPerIndexStats.incrementSuccess(indexA);
        RemoteIndexBuildPerIndexStats.incrementFailure(indexB);

        assertEquals(2L, RemoteIndexBuildPerIndexStats.getSuccessCount(indexA));
        assertEquals(0L, RemoteIndexBuildPerIndexStats.getFailureCount(indexA));
        assertEquals(0L, RemoteIndexBuildPerIndexStats.getSuccessCount(indexB));
        assertEquals(1L, RemoteIndexBuildPerIndexStats.getFailureCount(indexB));
    }

    public void testNullIndexNameIsIgnored() {
        // Must not throw; a null index name (defensive) is simply not recorded.
        RemoteIndexBuildPerIndexStats.incrementSuccess(null);
        RemoteIndexBuildPerIndexStats.incrementFailure(null);
        assertEquals(0L, RemoteIndexBuildPerIndexStats.getSuccessCount(null));
        assertEquals(0L, RemoteIndexBuildPerIndexStats.getFailureCount(null));
    }

    /**
     * The stats map exposed under remote_vector_index_build_stats.per_index.&lt;index&gt; must carry both
     * counts in the shape the integration test reads.
     */
    @SuppressWarnings("unchecked")
    public void testAsMapShapeAndValues() {
        String index = uniqueIndex();
        RemoteIndexBuildPerIndexStats.incrementSuccess(index);
        RemoteIndexBuildPerIndexStats.incrementSuccess(index);
        RemoteIndexBuildPerIndexStats.incrementFailure(index);

        Map<String, Object> map = RemoteIndexBuildPerIndexStats.asMap();
        assertTrue("per-index map must contain the index key", map.containsKey(index));

        Map<String, Object> perIndex = (Map<String, Object>) map.get(index);
        assertEquals(2L, perIndex.get(KNNRemoteIndexBuildValue.INDEX_BUILD_SUCCESS_COUNT.getName()));
        assertEquals(1L, perIndex.get(KNNRemoteIndexBuildValue.INDEX_BUILD_FAILURE_COUNT.getName()));
    }

    /**
     * Concurrency: increments from many threads (the real build path runs on flush/merge threads) must not
     * lose updates. LongAdder guarantees this; the test guards against a regression to a non-atomic map.
     */
    public void testConcurrentIncrementsAreNotLost() throws InterruptedException {
        final String index = uniqueIndex();
        final int threads = 8;
        final int perThread = 1000;
        Thread[] pool = new Thread[threads];
        for (int t = 0; t < threads; t++) {
            pool[t] = new Thread(() -> {
                for (int i = 0; i < perThread; i++) {
                    RemoteIndexBuildPerIndexStats.incrementSuccess(index);
                }
            });
        }
        for (Thread th : pool) {
            th.start();
        }
        for (Thread th : pool) {
            th.join();
        }
        assertEquals((long) threads * perThread, RemoteIndexBuildPerIndexStats.getSuccessCount(index));
    }
}
