/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.SneakyThrows;
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
            Map.of(KNNSettingsDefinitions.KNN_MEMORY_CIRCUIT_BREAKER_LIMIT, "\"" + expectedKNNCircuitBreakerLimit + "kb\"")
        );
        mockNode.start();
        ClusterService clusterService = mockNode.injector().getInstance(ClusterService.class);
        KNNSettings.state().initialize(clusterService);
        long actualKNNCircuitBreakerLimit = ((ByteSizeValue) KNNSettings.state()
            .getSettingValue(KNNSettingsDefinitions.KNN_MEMORY_CIRCUIT_BREAKER_LIMIT)).getKb();
        mockNode.close();
        assertEquals(expectedKNNCircuitBreakerLimit, actualKNNCircuitBreakerLimit);
        assertWarnings();
    }

    @SneakyThrows
    public void testGetSettingValueDefault() {
        Node mockNode = createMockNode(Collections.emptyMap());
        mockNode.start();
        ClusterService clusterService = mockNode.injector().getInstance(ClusterService.class);
        KNNSettings.state().initialize(clusterService);
        long actualKNNCircuitBreakerLimit = ((ByteSizeValue) KNNSettings.state()
            .getSettingValue(KNNSettingsDefinitions.KNN_MEMORY_CIRCUIT_BREAKER_LIMIT)).getKb();
        mockNode.close();
        assertEquals(
            ((ByteSizeValue) KNNSettingsDefinitions.dynamicCacheSettings.get(KNNSettingsDefinitions.KNN_MEMORY_CIRCUIT_BREAKER_LIMIT)
                .getDefault(Settings.EMPTY)).getKb(),
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
        KNNSettings.state().initialize(clusterService);

        Integer filteredSearchThreshold = KNNSettings.getFilteredExactSearchThreshold(INDEX_NAME);
        mockNode.close();
        assertEquals(KNNSettingsDefinitions.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD_DEFAULT_VALUE, filteredSearchThreshold);
        assertWarnings();
    }

    @SneakyThrows
    public void testFilteredSearchAdvanceSetting_whenValuesProvidedByUsers_thenValidateSameValues() {
        int userDefinedThreshold = 1000;
        int userDefinedThresholdMinValue = 0;
        Node mockNode = createMockNode(Collections.emptyMap());
        mockNode.start();
        ClusterService clusterService = mockNode.injector().getInstance(ClusterService.class);
        mockNode.client().admin().cluster().state(new ClusterStateRequest()).actionGet();
        mockNode.client().admin().indices().create(new CreateIndexRequest(INDEX_NAME)).actionGet();
        KNNSettings.state().initialize(clusterService);

        final Settings filteredSearchAdvanceSettings = Settings.builder()
            .put(KNNSettingsDefinitions.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, userDefinedThreshold)
            .build();

        mockNode.client()
            .admin()
            .indices()
            .updateSettings(new UpdateSettingsRequest(filteredSearchAdvanceSettings, INDEX_NAME))
            .actionGet();

        int filteredSearchThreshold = KNNSettings.getFilteredExactSearchThreshold(INDEX_NAME);

        // validate if we are able to set MinValues for the setting
        final Settings filteredSearchAdvanceSettingsWithMinValues = Settings.builder()
            .put(KNNSettingsDefinitions.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, userDefinedThresholdMinValue)
            .build();

        mockNode.client()
            .admin()
            .indices()
            .updateSettings(new UpdateSettingsRequest(filteredSearchAdvanceSettingsWithMinValues, INDEX_NAME))
            .actionGet();

        int filteredSearchThresholdMinValue = KNNSettings.getFilteredExactSearchThreshold(INDEX_NAME);

        mockNode.close();
        assertEquals(userDefinedThreshold, filteredSearchThreshold);
        assertEquals(userDefinedThresholdMinValue, filteredSearchThresholdMinValue);
        assertWarnings();
    }

    @SneakyThrows
    public void testGetEfSearch_whenNoValuesProvidedByUsers_thenDefaultSettingsUsed() {
        Node mockNode = createMockNode(Collections.emptyMap());
        mockNode.start();
        ClusterService clusterService = mockNode.injector().getInstance(ClusterService.class);
        mockNode.client().admin().cluster().state(new ClusterStateRequest()).actionGet();
        mockNode.client().admin().indices().create(new CreateIndexRequest(INDEX_NAME)).actionGet();
        KNNSettings.state().initialize(clusterService);

        Integer efSearchValue = KNNSettings.getEfSearchParam(INDEX_NAME);
        mockNode.close();
        assertEquals(KNNSettingsDefinitions.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH, efSearchValue);
        assertWarnings();
    }

    @SneakyThrows
    public void testGetEfSearch_whenEFSearchValueSetByUser_thenReturnValue() {
        int userProvidedEfSearch = 300;
        Node mockNode = createMockNode(Collections.emptyMap());
        mockNode.start();
        ClusterService clusterService = mockNode.injector().getInstance(ClusterService.class);
        mockNode.client().admin().cluster().state(new ClusterStateRequest()).actionGet();
        final Settings settings = Settings.builder()
            .put(KNNSettingsDefinitions.KNN_ALGO_PARAM_EF_SEARCH, userProvidedEfSearch)
            .put(KNNSettingsDefinitions.KNN_INDEX, true)
            .build();
        mockNode.client().admin().indices().create(new CreateIndexRequest(INDEX_NAME, settings)).actionGet();
        KNNSettings.state().initialize(clusterService);

        int efSearchValue = KNNSettings.getEfSearchParam(INDEX_NAME);
        mockNode.close();
        assertEquals(userProvidedEfSearch, efSearchValue);
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
