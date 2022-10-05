/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import com.carrotsearch.hppc.cursors.ObjectCursor;
import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.opensearch.Version;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.collect.ImmutableOpenMap;

/**
 * Class abstracts information related to underlying OpenSearch cluster
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
@Log4j2
public class KNNClusterContext {

    private ClusterService clusterService;
    private static KNNClusterContext instance;

    /**
     * Return instance of the cluster context, must be initialized first for proper usage
     * @return instance of cluster context
     */
    public static synchronized KNNClusterContext instance() {
        if (instance == null) {
            instance = new KNNClusterContext();
        }
        return instance;
    }

    /**
     * Initializes instance of cluster context by injecting dependencies
     * @param clusterService
     */
    public void initialize(final ClusterService clusterService) {
        this.clusterService = clusterService;
    }

    /**
     * Return minimal OpenSearch version based on all nodes currently discoverable in the cluster
     * @return minimal installed OpenSearch version, default to Version.CURRENT which is typically the latest version
     */
    public Version getClusterMinVersion() {
        Version minVersion = Version.CURRENT;
        ImmutableOpenMap<String, DiscoveryNode> clusterDiscoveryNodes = ImmutableOpenMap.of();
        log.debug("Reading cluster min version");
        try {
            clusterDiscoveryNodes = this.clusterService.state().getNodes().getNodes();
        } catch (Exception exception) {
            log.error("Cannot get cluster nodes", exception);
        }
        for (final ObjectCursor<DiscoveryNode> discoveryNodeCursor : clusterDiscoveryNodes.values()) {
            final Version nodeVersion = discoveryNodeCursor.value.getVersion();
            if (nodeVersion.before(minVersion)) {
                minVersion = nodeVersion;
                log.debug("Update cluster min version to {} based on node {}", nodeVersion, discoveryNodeCursor.value.toString());
            }
        }
        log.debug("Return cluster min version {}", minVersion);
        return minVersion;
    }
}
