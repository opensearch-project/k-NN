/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.opensearch.common.collect.Tuple;
import org.opensearch.knn.KNNTestCase;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.ThreadPoolExecutor;

public class KNN1040PerFieldKnnVectorsFormatTests extends KNNTestCase {

    public void testToTinySegmentsThreshold_whenNegativeOne_thenReturnsIntegerMaxValue() {
        assertEquals(Integer.MAX_VALUE, KNN1040PerFieldKnnVectorsFormat.toTinySegmentsThreshold(-1));
    }

    public void testToTinySegmentsThreshold_whenZero_thenReturnsZero() {
        // 0 → 0 → docCount < 0 is never true → always build the graph (matches Faiss semantics).
        assertEquals(0, KNN1040PerFieldKnnVectorsFormat.toTinySegmentsThreshold(0));
    }

    public void testToTinySegmentsThreshold_whenPositive_thenReturnsSameValue() {
        assertEquals(500, KNN1040PerFieldKnnVectorsFormat.toTinySegmentsThreshold(500));
        assertEquals(100, KNN1040PerFieldKnnVectorsFormat.toTinySegmentsThreshold(100));
        assertEquals(1, KNN1040PerFieldKnnVectorsFormat.toTinySegmentsThreshold(1));
    }

    public void testToTinySegmentsThreshold_whenLargeNegative_thenReturnsIntegerMaxValue() {
        assertEquals(Integer.MAX_VALUE, KNN1040PerFieldKnnVectorsFormat.toTinySegmentsThreshold(Integer.MIN_VALUE));
    }

    public void testToTinySegmentsThreshold_whenIntegerMaxValue_thenReturnsSameValue() {
        assertEquals(Integer.MAX_VALUE, KNN1040PerFieldKnnVectorsFormat.toTinySegmentsThreshold(Integer.MAX_VALUE));
    }

    public void testBuildMergeThreadCountAndExecutorService_whenThreadQtyIsOne_thenReturnsNullExecutor() {
        Tuple<Integer, ExecutorService> result = KNN1040PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(1);
        assertEquals(Integer.valueOf(1), result.v1());
        assertNull(result.v2());
    }

    public void testBuildMergeThreadCountAndExecutorService_whenThreadQtyIsZero_thenReturnsNullExecutor() {
        Tuple<Integer, ExecutorService> result = KNN1040PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(0);
        assertEquals(Integer.valueOf(1), result.v1());
        assertNull(result.v2());
    }

    public void testBuildMergeThreadCountAndExecutorService_whenThreadQtyIsNegative_thenReturnsNullExecutor() {
        Tuple<Integer, ExecutorService> result = KNN1040PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(-1);
        assertEquals(Integer.valueOf(1), result.v1());
        assertNull(result.v2());
    }

    public void testBuildMergeThreadCountAndExecutorService_whenThreadQtyAboveOne_thenReturnsExecutorWithCorrectPoolSize() {
        Tuple<Integer, ExecutorService> result = KNN1040PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(4);
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
        Tuple<Integer, ExecutorService> result = KNN1040PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(2);
        try {
            ThreadPoolExecutor executor = (ThreadPoolExecutor) result.v2();
            assertTrue(executor.allowsCoreThreadTimeOut());
        } finally {
            result.v2().shutdownNow();
        }
    }

    public void testBuildMergeThreadCountAndExecutorService_whenCalledMultipleTimes_thenReturnsIndependentExecutors() {
        Tuple<Integer, ExecutorService> first = KNN1040PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(4);
        Tuple<Integer, ExecutorService> second = KNN1040PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(4);
        try {
            assertNotSame(first.v2(), second.v2());
        } finally {
            first.v2().shutdownNow();
            second.v2().shutdownNow();
        }
    }
}
