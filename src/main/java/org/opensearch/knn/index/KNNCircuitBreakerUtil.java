/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.extern.log4j.Log4j2;
import org.opensearch.action.admin.cluster.settings.ClusterUpdateSettingsRequest;
import org.opensearch.action.admin.cluster.settings.ClusterUpdateSettingsResponse;
import org.opensearch.client.Client;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.core.action.ActionListener;
import org.opensearch.knn.plugin.stats.StatNames;
import org.opensearch.knn.plugin.transport.KNNStatsAction;
import org.opensearch.knn.plugin.transport.KNNStatsNodeResponse;
import org.opensearch.knn.plugin.transport.KNNStatsRequest;
import org.opensearch.knn.plugin.transport.KNNStatsResponse;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

import static org.opensearch.knn.index.KNNSettingsDefinitions.KNN_CIRCUIT_BREAKER_TRIGGERED;

/**
 * Singleton utility class to interact with the circuit breaker.
 */
@Log4j2
public class KNNCircuitBreakerUtil {
    private static KNNCircuitBreakerUtil instance;

    private KNNCircuitBreakerUtil() {}

    private Client client;

    /**
     * Return instance of the KNNCircuitBreakerUtil, must be initialized first for proper usage
     * @return instance of KNNCircuitBreakerUtil
     */
    public static synchronized KNNCircuitBreakerUtil instance() {
        if (instance == null) {
            instance = new KNNCircuitBreakerUtil();
        }
        return instance;
    }

    /**
     *
     * @param client client for interfacing with the cluster
     */
    public void initialize(final Client client) {
        this.client = client;
    }

    /**
     * Get the nodes in cluster that have reached the maximum capacity
     *
     * @return list of IDs for nodes that have reached the max capacity
     */
    public List<String> getNodesAtMaxCapacity() throws ExecutionException, InterruptedException {
        KNNStatsRequest knnStatsRequest = new KNNStatsRequest();
        knnStatsRequest.addStat(StatNames.CACHE_CAPACITY_REACHED.getName());
        knnStatsRequest.timeout(new TimeValue(1000 * 10)); // 10 second timeout

        KNNStatsResponse knnStatsResponse = client.execute(KNNStatsAction.INSTANCE, knnStatsRequest).get();
        List<KNNStatsNodeResponse> nodeResponses = knnStatsResponse.getNodes();

        List<String> nodesAtMaxCapacity = new ArrayList<>();
        for (KNNStatsNodeResponse nodeResponse : nodeResponses) {
            if ((Boolean) nodeResponse.getStatsMap().get(StatNames.CACHE_CAPACITY_REACHED.getName())) {
                nodesAtMaxCapacity.add(nodeResponse.getNode().getId());
            }
        }

        return nodesAtMaxCapacity;
    }

    /**
     * Updates knn.circuit_breaker.triggered setting to true/false
     * @param flag true/false
     */
    public synchronized void updateCircuitBreakerSettings(boolean flag) {
        ClusterUpdateSettingsRequest clusterUpdateSettingsRequest = new ClusterUpdateSettingsRequest();
        Settings circuitBreakerSettings = Settings.builder().put(KNN_CIRCUIT_BREAKER_TRIGGERED, flag).build();
        clusterUpdateSettingsRequest.persistentSettings(circuitBreakerSettings);
        client.admin().cluster().updateSettings(clusterUpdateSettingsRequest, new ActionListener<>() {
            @Override
            public void onResponse(ClusterUpdateSettingsResponse clusterUpdateSettingsResponse) {
                log.debug(
                    "Cluster setting {}, acknowledged: {} ",
                    clusterUpdateSettingsRequest.persistentSettings(),
                    clusterUpdateSettingsResponse.isAcknowledged()
                );
            }

            @Override
            public void onFailure(Exception e) {
                log.info(
                    "Exception while updating circuit breaker setting {} to {}",
                    clusterUpdateSettingsRequest.persistentSettings(),
                    e.getMessage()
                );
            }
        });
    }
}
