/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.backward_codecs.KNN9120Codec;

import org.opensearch.common.collect.Tuple;
import org.opensearch.knn.KNNTestCase;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.ThreadPoolExecutor;

public class KNN9120PerFieldKnnVectorsFormatTests extends KNNTestCase {

    public void testBuildMergeThreadCountAndExecutorService_whenThreadQtyIsOne_thenReturnsNullExecutor() {
        Tuple<Integer, ExecutorService> result = KNN9120PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(1);
        assertEquals(Integer.valueOf(1), result.v1());
        assertNull(result.v2());
    }

    public void testBuildMergeThreadCountAndExecutorService_whenThreadQtyIsZero_thenReturnsNullExecutor() {
        Tuple<Integer, ExecutorService> result = KNN9120PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(0);
        assertEquals(Integer.valueOf(1), result.v1());
        assertNull(result.v2());
    }

    public void testBuildMergeThreadCountAndExecutorService_whenThreadQtyIsNegative_thenReturnsNullExecutor() {
        Tuple<Integer, ExecutorService> result = KNN9120PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(-1);
        assertEquals(Integer.valueOf(1), result.v1());
        assertNull(result.v2());
    }

    public void testBuildMergeThreadCountAndExecutorService_whenThreadQtyAboveOne_thenReturnsExecutorWithCorrectPoolSize() {
        Tuple<Integer, ExecutorService> result = KNN9120PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(4);
        try {
            assertEquals(Integer.valueOf(4), result.v1());
            assertNotNull(result.v2());
            assertTrue(result.v2() instanceof ThreadPoolExecutor);
            ThreadPoolExecutor executor = (ThreadPoolExecutor) result.v2();
            assertEquals(4, executor.getCorePoolSize());
            assertEquals(4, executor.getMaximumPoolSize());
        } finally {
            result.v2().shutdownNow();
        }
    }

    public void testBuildMergeThreadCountAndExecutorService_whenThreadQtyAboveOne_thenExecutorAllowsCoreThreadTimeout() {
        Tuple<Integer, ExecutorService> result = KNN9120PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(2);
        try {
            ThreadPoolExecutor executor = (ThreadPoolExecutor) result.v2();
            assertTrue(executor.allowsCoreThreadTimeOut());
        } finally {
            result.v2().shutdownNow();
        }
    }

    public void testBuildMergeThreadCountAndExecutorService_whenCalledMultipleTimes_thenReturnsIndependentExecutors() {
        Tuple<Integer, ExecutorService> first = KNN9120PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(4);
        Tuple<Integer, ExecutorService> second = KNN9120PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(4);
        try {
            assertNotSame(first.v2(), second.v2());
        } finally {
            first.v2().shutdownNow();
            second.v2().shutdownNow();
        }
    }
}
