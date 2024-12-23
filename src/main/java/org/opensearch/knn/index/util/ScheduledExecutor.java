/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.util;

import lombok.Getter;

import java.io.Closeable;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Executes a task periodically
 */
public class ScheduledExecutor implements Closeable {
    final ScheduledExecutorService executor;
    @Getter
    private final Runnable task;

    /**
     * @param task task to be completed
     * @param scheduleMillis time in milliseconds to wait before executing the task again
     */
    public ScheduledExecutor(ScheduledExecutorService executor, Runnable task, long scheduleMillis) {
        this.task = task;
        this.executor = executor;
        executor.scheduleAtFixedRate(task, 0, scheduleMillis, TimeUnit.MILLISECONDS);
    }

    @Override
    public void close() {
        executor.shutdown();
        try {
            if (!executor.awaitTermination(60, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }

}
