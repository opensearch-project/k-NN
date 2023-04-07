/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common;

import org.opensearch.client.Client;
import org.opensearch.common.util.concurrent.ThreadContext;

import java.util.function.Supplier;

/**
 * Class abstracts execution of runnable or function in specific context
 */
public class TaskRunner {

    /**
     * Sets the thread context to default and execute function, this needed to allow actions on model system index
     * when security plugin is enabled
     * @param function runnable that needs to be executed after thread context has been stashed, accepts and returns nothing
     */
    public static void runWithStashedThreadContext(Client client, Runnable function) {
        try (ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()) {
            function.run();
        }
    }

    /**
     * Sets the thread context to default and execute function, this needed to allow actions on model system index
     * when security plugin is enabled
     * @param function supplier function that needs to be executed after thread context has been stashed, return object
     */
    public static <T> T runWithStashedThreadContext(Client client, Supplier<T> function) {
        try (ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()) {
            return function.get();
        }
    }
}
