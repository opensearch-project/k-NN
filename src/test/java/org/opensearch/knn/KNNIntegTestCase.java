/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn;

import org.junit.Before;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.knn.plugin.stats.KNNCounter;
import org.opensearch.plugins.Plugin;
import org.opensearch.test.OpenSearchIntegTestCase;

import java.util.Collection;
import java.util.Collections;

public class KNNIntegTestCase extends OpenSearchIntegTestCase {

    @Before
    public void setUp() throws Exception {
        internalCluster().startNodes(2);
        for (KNNCounter knnCounter : KNNCounter.values()) {
            knnCounter.set(0L);
        }
    }

    @Override
    protected Collection<Class<? extends Plugin>> nodePlugins() {
        return Collections.singletonList(KNNPlugin.class);
    }
}
