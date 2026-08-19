/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.Builder;
import org.opensearch.common.annotation.ExperimentalApi;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.transport.client.Client;

/**
 * Node services handed to a {@link KNNEngineDefinition} when discovery runs from the plugin lifecycle.
 * Every field is null when discovery runs outside a node (unit tests and tools). Constructed through the
 * builder so future services can be added without breaking definitions compiled against an older shape.
 */
@ExperimentalApi
@Builder
public record KNNEngineContext(Client client, ClusterService clusterService) {

    /** The context used when discovery runs outside a node. */
    public static final KNNEngineContext EMPTY = KNNEngineContext.builder().build();
}
