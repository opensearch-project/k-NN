/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.remote;

import lombok.extern.log4j.Log4j2;
import org.opensearch.common.StopWatch;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.io.IOException;

import static org.opensearch.knn.index.codec.util.KNNCodecUtil.initializeVectorValues;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.BUILD_REQUEST_FAILURE_COUNT;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.BUILD_REQUEST_SUCCESS_COUNT;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.INDEX_BUILD_FAILURE_COUNT;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.INDEX_BUILD_SUCCESS_COUNT;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.READ_FAILURE_COUNT;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.READ_SUCCESS_COUNT;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.READ_TIME;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.REMOTE_INDEX_BUILD_CURRENT_FLUSH_OPERATIONS;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.REMOTE_INDEX_BUILD_CURRENT_FLUSH_SIZE;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.REMOTE_INDEX_BUILD_CURRENT_MERGE_OPERATIONS;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.REMOTE_INDEX_BUILD_CURRENT_MERGE_SIZE;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.REMOTE_INDEX_BUILD_FLUSH_TIME;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.REMOTE_INDEX_BUILD_MERGE_TIME;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.WAITING_TIME;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.WRITE_FAILURE_COUNT;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.WRITE_SUCCESS_COUNT;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.WRITE_TIME;

/**
 * Class to handle all metric collection for the remote index build.
 * Each phase has its own StopWatch and `start` and `end` methods.
 */
@Log4j2
public class RemoteIndexBuildMetrics {
    private final StopWatch overallStopWatch;
    private final StopWatch writeStopWatch;
    private final StopWatch buildRequestStopWatch;
    private final StopWatch waiterStopWatch;
    private final StopWatch readStopWatch;
    private long size;
    private boolean isFlush;
    private String fieldName;

    public RemoteIndexBuildMetrics() {
        this.overallStopWatch = new StopWatch();
        this.writeStopWatch = new StopWatch();
        this.buildRequestStopWatch = new StopWatch();
        this.waiterStopWatch = new StopWatch();
        this.readStopWatch = new StopWatch();
    }

    /**
     * Helper method to collect remote index build metrics on start
     */
    public void startRemoteIndexBuildMetrics(BuildIndexParams indexInfo) throws IOException {
        KNNVectorValues<?> knnVectorValues = indexInfo.getKnnVectorValuesSupplier().get();
        initializeVectorValues(knnVectorValues);
        this.size = (long) indexInfo.getTotalLiveDocs() * knnVectorValues.bytesPerVector();
        this.isFlush = indexInfo.isFlush();
        this.fieldName = indexInfo.getFieldName();
        overallStopWatch.start();
        if (isFlush) {
            REMOTE_INDEX_BUILD_CURRENT_FLUSH_OPERATIONS.increment();
            REMOTE_INDEX_BUILD_CURRENT_FLUSH_SIZE.incrementBy(size);
        } else {
            REMOTE_INDEX_BUILD_CURRENT_MERGE_OPERATIONS.increment();
            REMOTE_INDEX_BUILD_CURRENT_MERGE_SIZE.incrementBy(size);
        }
    }

    // Repository read phase metric helpers
    public void startRepositoryWriteMetrics() {
        writeStopWatch.start();
    }

    public void endRepositoryWriteMetrics(boolean success) {
        long time_in_millis = writeStopWatch.stop().totalTime().millis();
        if (success) {
            WRITE_SUCCESS_COUNT.increment();
            WRITE_TIME.incrementBy(time_in_millis);
            log.debug("Repository write took {} ms for vector field [{}]", time_in_millis, fieldName);
        } else {
            WRITE_FAILURE_COUNT.increment();
        }
    }

    // Build request phase metric helpers
    public void startBuildRequestMetrics() {
        buildRequestStopWatch.start();
    }

    public void endBuildRequestMetrics(boolean success) {
        long time_in_millis = buildRequestStopWatch.stop().totalTime().millis();
        if (success) {
            BUILD_REQUEST_SUCCESS_COUNT.increment();
            log.debug("Submit vector build took {} ms for vector field [{}]", time_in_millis, fieldName);
        } else {
            BUILD_REQUEST_FAILURE_COUNT.increment();
        }
    }

    // Await index build phase metric helpers
    public void startWaitingMetrics() {
        waiterStopWatch.start();
    }

    public void endWaitingMetrics() {
        long time_in_millis = waiterStopWatch.stop().totalTime().millis();
        WAITING_TIME.incrementBy(time_in_millis);
        log.debug("Await vector build took {} ms for vector field [{}]", time_in_millis, fieldName);
    }

    // Repository read phase metric helpers
    public void startRepositoryReadMetrics() {
        readStopWatch.start();
    }

    public void endRepositoryReadMetrics(boolean success) {
        long time_in_millis = readStopWatch.stop().totalTime().millis();
        if (success) {
            READ_SUCCESS_COUNT.increment();
            READ_TIME.incrementBy(time_in_millis);
            log.debug("Repository read took {} ms for vector field [{}]", time_in_millis, fieldName);
        } else {
            READ_FAILURE_COUNT.increment();
        }
    }

    /**
     * Helper method to collect overall remote index build metrics
     */
    public void endRemoteIndexBuildMetrics(boolean wasSuccessful) {
        long time_in_millis = overallStopWatch.stop().totalTime().millis();
        if (wasSuccessful) {
            INDEX_BUILD_SUCCESS_COUNT.increment();
            log.debug("Remote index build succeeded after {} ms for vector field [{}]", time_in_millis, fieldName);
        } else {
            INDEX_BUILD_FAILURE_COUNT.increment();
            log.debug("Remote index build failed after {} ms for vector field [{}]", time_in_millis, fieldName);
        }
        if (isFlush) {
            REMOTE_INDEX_BUILD_CURRENT_FLUSH_OPERATIONS.decrement();
            REMOTE_INDEX_BUILD_CURRENT_FLUSH_SIZE.decrementBy(size);
            REMOTE_INDEX_BUILD_FLUSH_TIME.incrementBy(time_in_millis);
        } else {
            REMOTE_INDEX_BUILD_CURRENT_MERGE_OPERATIONS.decrement();
            REMOTE_INDEX_BUILD_CURRENT_MERGE_SIZE.decrementBy(size);
            REMOTE_INDEX_BUILD_MERGE_TIME.incrementBy(time_in_millis);
        }
    }
}
