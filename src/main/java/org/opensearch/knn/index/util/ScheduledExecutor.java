/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.util;

import java.io.Closeable;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Executes a task periodically
 */
public class ScheduledExecutor implements Closeable {
    final ScheduledExecutorService executor;
    public final Runnable task;

    /**
     * @param task task to be completed
     * @param scheduleMillis time in milliseconds to wait before executing the task again
     */
    public ScheduledExecutor(Runnable task, long scheduleMillis) {
        this.task = task;
        this.executor = Executors.newSingleThreadScheduledExecutor();
        executor.scheduleAtFixedRate(task, 0, scheduleMillis, TimeUnit.MILLISECONDS);
    }

    @Override
    public void close() {
        executor.shutdown();
    }
}
