/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.SneakyThrows;
import org.junit.Assert;
import org.opensearch.action.admin.cluster.state.ClusterStateRequest;
import org.opensearch.action.admin.indices.create.CreateIndexRequest;
import org.opensearch.action.admin.indices.settings.put.UpdateSettingsRequest;
import org.opensearch.cluster.ClusterName;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.network.NetworkModule;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.env.Environment;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.node.MockNode;
import org.opensearch.node.Node;
import org.opensearch.plugins.Plugin;
import org.opensearch.test.InternalTestCluster;
import org.opensearch.test.MockHttpTransport;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.opensearch.test.NodeRoles.dataNode;

public class KNNSettingsTests extends KNNTestCase {

    private static final String INDEX_NAME = "myindex";

    @SneakyThrows
    public void testGetSettingValueFromConfig() {
        long expectedKNNCircuitBreakerLimit = 13;
        Node mockNode = createMockNode(
            Map.of(KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_LIMIT, "\"" + expectedKNNCircuitBreakerLimit + "kb\"")
        );
        mockNode.start();
        ClusterService clusterService = mockNode.injector().getInstance(ClusterService.class);
        KNNSettings.state().setClusterService(clusterService);
        long actualKNNCircuitBreakerLimit = ((ByteSizeValue) KNNSettings.state()
            .getSettingValue(KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_LIMIT)).getKb();
        mockNode.close();
        assertEquals(expectedKNNCircuitBreakerLimit, actualKNNCircuitBreakerLimit);
        assertWarnings();
    }

    @SneakyThrows
    public void testGetSettingValueDefault() {
        Node mockNode = createMockNode(Collections.emptyMap());
        mockNode.start();
        ClusterService clusterService = mockNode.injector().getInstance(ClusterService.class);
        KNNSettings.state().setClusterService(clusterService);
        long actualKNNCircuitBreakerLimit = ((ByteSizeValue) KNNSettings.state()
            .getSettingValue(KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_LIMIT)).getKb();
        mockNode.close();
        assertEquals(
            ((ByteSizeValue) KNNSettings.dynamicCacheSettings.get(KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_LIMIT).getDefault(Settings.EMPTY))
                .getKb(),
            actualKNNCircuitBreakerLimit

        );
        // set warning for deprecation of index.store.hybrid.mmap.extensions as expected temporarily, need to work on proper strategy of
        // switching to new setting in core
        // no-jdk distributions expected warning is a workaround for running tests locally
        assertWarnings();
    }

    @SneakyThrows
    public void testFilteredSearchAdvanceSetting_whenNoValuesProvidedByUsers_thenDefaultSettingsUsed() {
        Node mockNode = createMockNode(Collections.emptyMap());
        mockNode.start();
        ClusterService clusterService = mockNode.injector().getInstance(ClusterService.class);
        mockNode.client().admin().cluster().state(new ClusterStateRequest()).actionGet();
        mockNode.client().admin().indices().create(new CreateIndexRequest(INDEX_NAME)).actionGet();
        KNNSettings.state().setClusterService(clusterService);

        int filteredSearchThresholdPct = KNNSettings.getFilteredExactSearchThresholdPct(INDEX_NAME);
        int filteredSearchThreshold = KNNSettings.getFilteredExactSearchThreshold(INDEX_NAME);
        mockNode.close();
        assertEquals((int) KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD_PCT_DEFAULT_VALUE, filteredSearchThresholdPct);
        assertEquals((int) KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD_DEFAULT_VALUE, filteredSearchThreshold);
        assertWarnings();
    }

    @SneakyThrows
    public void testFilteredSearchAdvanceSetting_whenValuesProvidedByUsers_thenValidateSameValues() {
        int userDefinedPctThreshold = 20;
        int userDefinedThreshold = 1000;
        int userDefinedPctThresholdMinValue = 0;
        int userDefinedThresholdMinValue = 0;
        Node mockNode = createMockNode(Collections.emptyMap());
        mockNode.start();
        ClusterService clusterService = mockNode.injector().getInstance(ClusterService.class);
        mockNode.client().admin().cluster().state(new ClusterStateRequest()).actionGet();
        mockNode.client().admin().indices().create(new CreateIndexRequest(INDEX_NAME)).actionGet();
        KNNSettings.state().setClusterService(clusterService);

        final Settings filteredSearchAdvanceSettings = Settings.builder()
            .put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, userDefinedThreshold)
            .put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD_PCT, userDefinedPctThreshold)
            .build();

        mockNode.client()
            .admin()
            .indices()
            .updateSettings(new UpdateSettingsRequest(filteredSearchAdvanceSettings, INDEX_NAME))
            .actionGet();

        int filteredSearchThresholdPct = KNNSettings.getFilteredExactSearchThresholdPct(INDEX_NAME);
        int filteredSearchThreshold = KNNSettings.getFilteredExactSearchThreshold(INDEX_NAME);

        // validate if we are able to set MinValues for the setting
        final Settings filteredSearchAdvanceSettingsWithMinValues = Settings.builder()
            .put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, userDefinedThresholdMinValue)
            .put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD_PCT, userDefinedPctThresholdMinValue)
            .build();

        mockNode.client()
            .admin()
            .indices()
            .updateSettings(new UpdateSettingsRequest(filteredSearchAdvanceSettingsWithMinValues, INDEX_NAME))
            .actionGet();

        int filteredSearchThresholdPctMinValue = KNNSettings.getFilteredExactSearchThresholdPct(INDEX_NAME);
        int filteredSearchThresholdMinValue = KNNSettings.getFilteredExactSearchThreshold(INDEX_NAME);

        // Validate if less than MinValues are set then Exception Happens
        final Settings filteredSearchAdvanceSettingsWithLessThanMinValues = Settings.builder()
            .put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, -1)
            .put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD_PCT, -1)
            .build();

        Assert.assertThrows(
            IllegalArgumentException.class,
            () -> mockNode.client()
                .admin()
                .indices()
                .updateSettings(new UpdateSettingsRequest(filteredSearchAdvanceSettingsWithLessThanMinValues, INDEX_NAME))
                .actionGet()
        );

        mockNode.close();
        assertEquals(userDefinedPctThreshold, filteredSearchThresholdPct);
        assertEquals(userDefinedThreshold, filteredSearchThreshold);
        assertEquals(userDefinedPctThresholdMinValue, filteredSearchThresholdPctMinValue);
        assertEquals(userDefinedThresholdMinValue, filteredSearchThresholdMinValue);
        assertWarnings();
    }

    private Node createMockNode(Map<String, Object> configSettings) throws IOException {
        Path configDir = createTempDir();
        File configFile = configDir.resolve("opensearch.yml").toFile();
        FileWriter configFileWriter = new FileWriter(configFile);

        for (Map.Entry<String, Object> setting : configSettings.entrySet()) {
            configFileWriter.write("\"" + setting.getKey() + "\": " + setting.getValue());
        }
        configFileWriter.close();
        return new MockNode(baseSettings().build(), basePlugins(), configDir, true);
    }

    private List<Class<? extends Plugin>> basePlugins() {
        List<Class<? extends Plugin>> plugins = new ArrayList<>();
        plugins.add(getTestTransportPlugin());
        plugins.add(MockHttpTransport.TestPlugin.class);
        plugins.add(KNNPlugin.class);
        return plugins;
    }

    private static Settings.Builder baseSettings() {
        final Path tempDir = createTempDir();
        return Settings.builder()
            .put(ClusterName.CLUSTER_NAME_SETTING.getKey(), InternalTestCluster.clusterName("single-node-cluster", randomLong()))
            .put(Environment.PATH_HOME_SETTING.getKey(), tempDir)
            .put(NetworkModule.TRANSPORT_TYPE_KEY, getTestTransportType())
            .put(dataNode());
    }

    private void assertWarnings() {
        // set warning for deprecation of index.store.hybrid.mmap.extensions as expected temporarily, need to work on proper strategy of
        // switching to new setting in core
        // no-jdk distributions expected warning is a workaround for running tests locally
        assertWarnings(
            "[index.store.hybrid.mmap.extensions] setting was deprecated in OpenSearch and will be removed in a future release! See the breaking changes documentation for the next major version.",
            "no-jdk distributions that do not bundle a JDK are deprecated and will be removed in a future release"
        );
    }
}
