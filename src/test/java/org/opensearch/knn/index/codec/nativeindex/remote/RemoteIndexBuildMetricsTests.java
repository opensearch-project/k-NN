/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.remote;

import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.codec.nativeindex.remote.RemoteIndexBuildStrategy.BuildResult;
import org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue;

import java.io.IOException;

import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.INDEX_BUILD_FAILURE_COUNT;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.INDEX_BUILD_MERGE_ABORT_EXCEPTION;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.INDEX_BUILD_SUCCESS_COUNT;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.INDEX_BUILD_TERMINAL_EXCEPTION;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.REMOTE_INDEX_BUILD_CURRENT_FLUSH_OPERATIONS;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.REMOTE_INDEX_BUILD_CURRENT_FLUSH_SIZE;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.REMOTE_INDEX_BUILD_CURRENT_MERGE_OPERATIONS;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.REMOTE_INDEX_BUILD_CURRENT_MERGE_SIZE;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.REMOTE_INDEX_BUILD_FLUSH_TIME;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.REMOTE_INDEX_BUILD_MERGE_TIME;

/**
 * Unit tests for {@link RemoteIndexBuildMetrics}, focusing on {@link RemoteIndexBuildMetrics#endRemoteIndexBuildMetrics}
 * which maps each {@link BuildResult} to the correct outcome counter. Benign terminations (merge aborts, terminal IO)
 * must be tracked with their own counters rather than {@link KNNRemoteIndexBuildValue#INDEX_BUILD_FAILURE_COUNT}.
 */
public class RemoteIndexBuildMetricsTests extends RemoteIndexBuildTests {

    /**
     * A SUCCESS outcome increments only the success counter.
     */
    public void testEndMetricsSuccessIncrementsSuccessCounterOnly() throws IOException {
        runAndAssertOutcome(BuildResult.SUCCESS, INDEX_BUILD_SUCCESS_COUNT);
    }

    /**
     * A FAILURE outcome increments only the failure counter.
     */
    public void testEndMetricsFailureIncrementsFailureCounterOnly() throws IOException {
        runAndAssertOutcome(BuildResult.FAILURE, INDEX_BUILD_FAILURE_COUNT);
    }

    /**
     * A MERGE_ABORT is a benign cancellation, tracked by its own counter and not counted as a build failure.
     */
    public void testEndMetricsMergeAbortIncrementsAbortCounterOnly() throws IOException {
        runAndAssertOutcome(BuildResult.MERGE_ABORT, INDEX_BUILD_MERGE_ABORT_EXCEPTION);
    }

    /**
     * A TERMINAL_IO termination is benign, tracked by its own counter and not counted as a build failure.
     */
    public void testEndMetricsTerminalIoIncrementsTerminalCounterOnly() throws IOException {
        runAndAssertOutcome(BuildResult.TERMINAL_IO, INDEX_BUILD_TERMINAL_EXCEPTION);
    }

    /**
     * For a flush operation, the current-flush gauges return to zero after start+end and only the flush timer is updated.
     */
    public void testEndMetricsFlushUpdatesFlushTimerAndGauges() throws IOException {
        RemoteIndexBuildMetrics metrics = new RemoteIndexBuildMetrics();
        metrics.startRemoteIndexBuildMetrics(buildParamsWithFlush(true));
        metrics.endRemoteIndexBuildMetrics(BuildResult.SUCCESS);

        // Start increments the gauges, end decrements them, so they net back to zero.
        assertEquals(0L, (long) REMOTE_INDEX_BUILD_CURRENT_FLUSH_OPERATIONS.getValue());
        assertEquals(0L, (long) REMOTE_INDEX_BUILD_CURRENT_FLUSH_SIZE.getValue());
        assertTrue(REMOTE_INDEX_BUILD_FLUSH_TIME.getValue() >= 0L);
        // Merge counters must remain untouched for a flush operation.
        assertEquals(0L, (long) REMOTE_INDEX_BUILD_CURRENT_MERGE_OPERATIONS.getValue());
        assertEquals(0L, (long) REMOTE_INDEX_BUILD_CURRENT_MERGE_SIZE.getValue());
        assertEquals(0L, (long) REMOTE_INDEX_BUILD_MERGE_TIME.getValue());
    }

    /**
     * For a merge operation, the current-merge gauges return to zero after start+end and only the merge timer is updated.
     */
    public void testEndMetricsMergeUpdatesMergeTimerAndGauges() throws IOException {
        RemoteIndexBuildMetrics metrics = new RemoteIndexBuildMetrics();
        metrics.startRemoteIndexBuildMetrics(buildParamsWithFlush(false));
        metrics.endRemoteIndexBuildMetrics(BuildResult.SUCCESS);

        assertEquals(0L, (long) REMOTE_INDEX_BUILD_CURRENT_MERGE_OPERATIONS.getValue());
        assertEquals(0L, (long) REMOTE_INDEX_BUILD_CURRENT_MERGE_SIZE.getValue());
        assertTrue(REMOTE_INDEX_BUILD_MERGE_TIME.getValue() >= 0L);
        // Flush counters must remain untouched for a merge operation.
        assertEquals(0L, (long) REMOTE_INDEX_BUILD_CURRENT_FLUSH_OPERATIONS.getValue());
        assertEquals(0L, (long) REMOTE_INDEX_BUILD_CURRENT_FLUSH_SIZE.getValue());
        assertEquals(0L, (long) REMOTE_INDEX_BUILD_FLUSH_TIME.getValue());
    }

    /**
     * Runs a full start/end metrics cycle for the given {@link BuildResult} and asserts that only {@code expectedCounter}
     * among the four outcome counters was incremented.
     */
    private void runAndAssertOutcome(BuildResult buildResult, KNNRemoteIndexBuildValue expectedCounter) throws IOException {
        RemoteIndexBuildMetrics metrics = new RemoteIndexBuildMetrics();
        metrics.startRemoteIndexBuildMetrics(buildIndexParams);
        metrics.endRemoteIndexBuildMetrics(buildResult);

        for (KNNRemoteIndexBuildValue outcomeCounter : new KNNRemoteIndexBuildValue[] {
            INDEX_BUILD_SUCCESS_COUNT,
            INDEX_BUILD_FAILURE_COUNT,
            INDEX_BUILD_MERGE_ABORT_EXCEPTION,
            INDEX_BUILD_TERMINAL_EXCEPTION }) {
            long expected = outcomeCounter == expectedCounter ? 1L : 0L;
            assertEquals("Unexpected value for " + outcomeCounter.getName(), expected, (long) outcomeCounter.getValue());
        }
    }

    /**
     * Builds a {@link BuildIndexParams} identical to the shared one but with a fixed {@code isFlush} value so the
     * flush vs. merge branches of {@link RemoteIndexBuildMetrics} can be exercised deterministically.
     */
    private BuildIndexParams buildParamsWithFlush(boolean isFlush) {
        return BuildIndexParams.builder()
            .indexOutputWithBuffer(indexOutputWithBuffer)
            .knnEngine(buildIndexParams.getKnnEngine())
            .field(buildIndexParams.getField())
            .vectorDataType(VectorDataType.FLOAT)
            .indexParameters(buildIndexParams.getIndexParameters())
            .knnVectorValuesSupplier(buildIndexParams.getKnnVectorValuesSupplier())
            .totalLiveDocs(buildIndexParams.getTotalLiveDocs())
            .segmentWriteState(segmentWriteState)
            .isFlush(isFlush)
            .build();
    }
}
