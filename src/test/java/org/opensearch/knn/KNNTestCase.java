/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn;

import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.unit.ByteSizeUnit;
import org.opensearch.common.unit.ByteSizeValue;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.memory.breaker.NativeMemoryCircuitBreakerService;
import org.opensearch.knn.plugin.stats.KNNCounter;
import org.opensearch.common.bytes.BytesReference;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.test.OpenSearchTestCase;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Base class for integration tests for KNN plugin. Contains several methods for testing KNN ES functionality.
 */
public class KNNTestCase extends OpenSearchTestCase {

    protected static final NativeMemoryCircuitBreakerService NEVER_TRIGGERED_CB_SERVICE = mock(NativeMemoryCircuitBreakerService.class);
    static {
        when(NEVER_TRIGGERED_CB_SERVICE.isCircuitBreakerTriggered()).thenReturn(false);
        when(NEVER_TRIGGERED_CB_SERVICE.getCircuitBreakerLimit()).thenReturn(new ByteSizeValue(100, ByteSizeUnit.KB));
    }

    @Mock
    protected ClusterService clusterService;
    private AutoCloseable openMocks;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        openMocks = MockitoAnnotations.openMocks(this);
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

        Set<Setting<?>> defaultClusterSettings = new HashSet<>(ClusterSettings.BUILT_IN_CLUSTER_SETTINGS);
        defaultClusterSettings.addAll(
            KNNSettings.state()
                .getSettings()
                .stream()
                .filter(s -> s.getProperties().contains(Setting.Property.NodeScope))
                .collect(Collectors.toList())
        );
        when(clusterService.getClusterSettings()).thenReturn(new ClusterSettings(Settings.EMPTY, defaultClusterSettings));
        KNNSettings.state().setClusterService(clusterService);

        // Clean up the cache
        NativeMemoryCacheManager.initialize(NEVER_TRIGGERED_CB_SERVICE);
        NativeMemoryCacheManager.getInstance().invalidateAll();
        NativeMemoryCacheManager.getInstance().close();
    }

    public Map<String, Object> xContentBuilderToMap(XContentBuilder xContentBuilder) {
        return XContentHelper.convertToMap(BytesReference.bytes(xContentBuilder), true, xContentBuilder.contentType()).v2();
    }
}
