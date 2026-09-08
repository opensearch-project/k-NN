/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.fixture;

import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNEngineContext;
import org.opensearch.knn.index.engine.KNNEngineDefinition;
import org.opensearch.knn.index.engine.KNNLibrary;

/**
 * Its initialize probes KNNEngine from ANOTHER thread and records whether the lookup returned while
 * discovery was still in flight. Pins that lookups never wait on the discovery lock, so a definition
 * that blocks in initialize cannot deadlock threads that only read engines.
 */
public final class ConcurrentLookupFixtureEngineProvider implements KNNEngineDefinition {

    static volatile boolean probeReturnedDuringInitialize = false;

    @Override
    public String engineName() {
        return "concurrent-lookup";
    }

    @Override
    public KNNLibrary library() {
        return PlainFixtureLibrary.INSTANCE;
    }

    @Override
    public void initialize(KNNEngineContext context) {
        final Thread probe = new Thread(() -> {
            if (KNNEngine.getEngine("faiss") == KNNEngine.FAISS) {
                probeReturnedDuringInitialize = true;
            }
        }, "fixture-concurrent-lookup-probe");
        probe.start();
        try {
            // Bounded join. If the probe were to park on the discovery lock this would time out and the
            // flag would stay false, failing the test instead of hanging the suite.
            probe.join(5_000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
