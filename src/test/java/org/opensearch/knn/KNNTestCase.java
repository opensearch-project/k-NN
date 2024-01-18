/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn;

import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.Metadata;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.knn.plugin.stats.KNNCounter;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.mockito.Mockito.when;

/**
 * Base class for integration tests for KNN plugin. Contains several methods for testing KNN ES functionality.
 */
public class KNNTestCase extends OpenSearchTestCase {

    @Mock
    protected ClusterService clusterService;
    @Mock
    protected ClusterState clusterState;
    @Mock
    protected Metadata metadata;
    @Mock
    protected IndexMetadata indexMetadata;
    @Mock
    protected KNNSettings knnSettings;
    private AutoCloseable openMocks;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        openMocks = MockitoAnnotations.openMocks(this);
        // This is required to make sure that before every test we are initializing the KNNSettings. Not doing this
        // leads to failures of unit tests cases when a unit test is run separately. Try running this test:
        // ./gradlew ':test' --tests "org.opensearch.knn.training.TrainingJobTests.testRun_success" and see it fails
        // but if run along with other tests this test passes.
        initKNNSettings();
    }

    @Override
    public void tearDown() throws Exception {
        super.tearDown();
        resetState();
        openMocks.close();
    }

    public void resetState() {
        // Reset all of the counters
        for (KNNCounter knnCounter : KNNCounter.values()) {
            knnCounter.set(0L);
        }
        initKNNSettings();

        // Clean up the cache
        NativeMemoryCacheManager.getInstance().invalidateAll();
        NativeMemoryCacheManager.getInstance().close();
    }

    private void initKNNSettings() {
        Set<Setting<?>> defaultClusterSettings = new HashSet<>(ClusterSettings.BUILT_IN_CLUSTER_SETTINGS);
        try (KNNPlugin plugin = new KNNPlugin()) {
            defaultClusterSettings.addAll(
                plugin.getSettings()
                    .stream()
                    .filter(s -> s.getProperties().contains(Setting.Property.NodeScope))
                    .collect(Collectors.toList())
            );
        } catch (IOException ioe) {
            fail("Failed to close KNNPlugin during initialization of settings");
        }

        when(clusterService.getClusterSettings()).thenReturn(new ClusterSettings(Settings.EMPTY, defaultClusterSettings));
        KNNSettings.state().initialize(clusterService);
    }

    public Map<String, Object> xContentBuilderToMap(XContentBuilder xContentBuilder) {
        return XContentHelper.convertToMap(BytesReference.bytes(xContentBuilder), true, xContentBuilder.contentType()).v2();
    }
}
