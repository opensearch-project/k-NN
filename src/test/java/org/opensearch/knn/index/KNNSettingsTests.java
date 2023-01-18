/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.SneakyThrows;
import org.opensearch.cluster.ClusterName;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.network.NetworkModule;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.unit.ByteSizeValue;
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
        assertEquals(actualKNNCircuitBreakerLimit, expectedKNNCircuitBreakerLimit);
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
            actualKNNCircuitBreakerLimit,
            ((ByteSizeValue) KNNSettings.dynamicCacheSettings.get(KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_LIMIT).getDefault(Settings.EMPTY))
                .getKb()
        );
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
}
