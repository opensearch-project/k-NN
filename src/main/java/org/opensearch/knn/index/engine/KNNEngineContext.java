/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.common.annotation.ExperimentalApi;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.transport.client.Client;

/**
 * Node services handed to a {@link KNNEngineDefinition} when discovery runs from the plugin lifecycle.
 * Both fields are null when discovery runs outside a node (unit tests and tools).
 */
@ExperimentalApi
public record KNNEngineContext(Client client, ClusterService clusterService) {

    /** The context used when discovery runs outside a node. */
    public static final KNNEngineContext EMPTY = new KNNEngineContext(null, null);
}
