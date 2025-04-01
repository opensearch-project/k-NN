/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.opensearch.common.settings.Settings;
import org.opensearch.knn.DerivedSourceTestCase;
import org.opensearch.knn.DerivedSourceUtils;
import org.opensearch.test.rest.OpenSearchRestTestCase;

import java.io.IOException;
import java.util.List;
import java.util.Optional;

import static org.opensearch.knn.TestUtils.BWC_VERSION;
import static org.opensearch.knn.TestUtils.CLIENT_TIMEOUT_VALUE;
import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;
import static org.opensearch.knn.TestUtils.RESTART_UPGRADE_OLD_CLUSTER;

public class DerivedSourceBWCRestartIT extends DerivedSourceTestCase {

    public void testFlat_indexAndForceMergeOnOld_injectOnNew() throws IOException {
        List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts = getFlatIndexContexts("knn-bwc", false, false);
        testIndexAndForceMergeOnOld_injectOnNew(indexConfigContexts);
    }

    public void testFlat_indexOnOld_forceMergeAndInjectOnNew() throws IOException {
        List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts = getFlatIndexContexts("knn-bwc", false, false);
        testIndexOnOld_forceMergeAndInjectOnNew(indexConfigContexts);
    }

    private void testIndexAndForceMergeOnOld_injectOnNew(List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts)
        throws IOException {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        if (isRunningAgainstOldCluster()) {
            prepareOriginalIndices(indexConfigContexts);
            testMerging(indexConfigContexts);

            // Delete
            testDelete(indexConfigContexts);
        } else {
            // Search
            testSearch(indexConfigContexts);

            // Reindex
            testReindex(indexConfigContexts);
        }
    }

    private void testIndexOnOld_forceMergeAndInjectOnNew(List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts)
        throws IOException {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        if (isRunningAgainstOldCluster()) {
            prepareOriginalIndices(indexConfigContexts);
        } else {
            testMerging(indexConfigContexts);

            // Delete
            testDelete(indexConfigContexts);
            // Search
            testSearch(indexConfigContexts);

            // Reindex
            testReindex(indexConfigContexts);
        }
    }

    @Override
    protected final boolean preserveIndicesUponCompletion() {
        return true;
    }

    @Override
    protected final boolean preserveReposUponCompletion() {
        return true;
    }

    @Override
    protected boolean preserveTemplatesUponCompletion() {
        return true;
    }

    @Override
    protected final Settings restClientSettings() {
        return Settings.builder()
            .put(super.restClientSettings())
            // increase the timeout here to 90 seconds to handle long waits for a green
            // cluster health. the waits for green need to be longer than a minute to
            // account for delayed shards
            .put(OpenSearchRestTestCase.CLIENT_SOCKET_TIMEOUT, CLIENT_TIMEOUT_VALUE)
            .build();
    }

    protected static final boolean isRunningAgainstOldCluster() {
        return Boolean.parseBoolean(System.getProperty(RESTART_UPGRADE_OLD_CLUSTER));
    }

    @Override
    protected final Optional<String> getBWCVersion() {
        return Optional.ofNullable(System.getProperty(BWC_VERSION, null));
    }
}
