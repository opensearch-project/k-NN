/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common;

import org.opensearch.client.node.NodeClient;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.threadpool.TestThreadPool;
import org.opensearch.threadpool.ThreadPool;

import java.util.function.Supplier;

public class ThreadContextHelperTests extends KNNTestCase {

    public void testRunWithStashedContextRunnable() {
        ThreadPool threadPool = new TestThreadPool(this.getClass().getSimpleName() + "ThreadPool");
        threadPool.getThreadContext().putHeader("key", "value");
        NodeClient client = new NodeClient(Settings.EMPTY, threadPool);

        assertTrue(client.threadPool().getThreadContext().getHeaders().containsKey("key"));

        Runnable runnable = () -> { assertFalse(client.threadPool().getThreadContext().getHeaders().containsKey("key")); };
        ThreadContextHelper.runWithStashedThreadContext(client, () -> runnable);

        assertTrue(client.threadPool().getThreadContext().getHeaders().containsKey("key"));

        threadPool.shutdownNow();
        client.close();
    }

    public void testRunWithStashedContextSupplier() {
        ThreadPool threadPool = new TestThreadPool(this.getClass().getSimpleName() + "ThreadPool");
        threadPool.getThreadContext().putHeader("key", "value");
        NodeClient client = new NodeClient(Settings.EMPTY, threadPool);

        assertTrue(client.threadPool().getThreadContext().getHeaders().containsKey("key"));

        Supplier<String> supplier = () -> {
            assertFalse(client.threadPool().getThreadContext().getHeaders().containsKey("key"));
            return this.getClass().getName();
        };
        ThreadContextHelper.runWithStashedThreadContext(client, () -> supplier);

        assertTrue(client.threadPool().getThreadContext().getHeaders().containsKey("key"));

        threadPool.shutdownNow();
        client.close();
    }
}
