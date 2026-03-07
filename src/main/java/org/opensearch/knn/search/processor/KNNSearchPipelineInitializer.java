/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.opensearch.action.search.PutSearchPipelineRequest;
import org.opensearch.action.support.clustermanager.AcknowledgedResponse;
import org.opensearch.cluster.ClusterChangedEvent;
import org.opensearch.cluster.ClusterStateListener;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.common.bytes.BytesArray;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.search.pipeline.SearchPipelineMetadata;
import org.opensearch.transport.client.Client;

import java.nio.charset.StandardCharsets;

import static org.opensearch.cluster.node.DiscoveryNode.isClusterManagerNode;

/**
 * Initializes the default KNN search pipeline that auto-excludes vector fields from _source.
 * The pipeline is created once when this node becomes the cluster manager. If creation fails,
 * it will be retried on the next cluster state change.
 */
@Log4j2
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class KNNSearchPipelineInitializer {

    public static final String KNN_DEFAULT_SEARCH_PIPELINE_NAME = "_knn_default_search_pipeline";

    private static final String PIPELINE_DEFINITION = "{"
            + "\"description\": \"Default search pipeline for KNN indices that excludes vector fields from _source\","
            + "\"request_processors\": [{\"" + KNNDefaultExcludesProcessor.TYPE + "\": {}}]"
            + "}";

    /**
     * Registers a ClusterStateListener that creates the default KNN search pipeline the first time
     * this node becomes the cluster manager. Retries on failure on subsequent cluster state changes.
     * The listener removes itself once the pipeline is confirmed in cluster state.
     */
    public static void initialize(Client client, ClusterService clusterService) {
        // Holding the reference to the listener in an array allows us to remove it later
        ClusterStateListener[] listenerRef = new ClusterStateListener[1];
        listenerRef[0] = (ClusterChangedEvent event) -> {
            if (!event.localNodeClusterManager()) {
                return;
            }
            SearchPipelineMetadata metadata = event.state().metadata().custom(SearchPipelineMetadata.TYPE);
            if (metadata != null && metadata.getPipelines().containsKey(KNN_DEFAULT_SEARCH_PIPELINE_NAME)) {
                clusterService.removeListener(listenerRef[0]);
                return;
            }
            client.admin().cluster().putSearchPipeline(
                    new PutSearchPipelineRequest(
                            KNN_DEFAULT_SEARCH_PIPELINE_NAME,
                            new BytesArray(PIPELINE_DEFINITION.getBytes(StandardCharsets.UTF_8)),
                            MediaTypeRegistry.JSON
                    ),
                    new ActionListener<>() {
                        @Override
                        public void onResponse(AcknowledgedResponse acknowledgedResponse) {
                            log.info("Created default KNN search pipeline [{}]", KNN_DEFAULT_SEARCH_PIPELINE_NAME);
                            clusterService.removeListener(listenerRef[0]);
                        }

                        @Override
                        public void onFailure(Exception e) {
                            log.warn("Failed to create default KNN search pipeline [{}]: {}", KNN_DEFAULT_SEARCH_PIPELINE_NAME, e.getMessage());
                        }
                    });
        };

        if (isClusterManagerNode(clusterService.getSettings())) {
            clusterService.addListener(listenerRef[0]);
        }
    }
}
