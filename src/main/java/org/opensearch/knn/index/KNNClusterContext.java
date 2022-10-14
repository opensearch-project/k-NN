/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.opensearch.Version;
import org.opensearch.cluster.service.ClusterService;

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
        try {
            minVersion = this.clusterService.state().getNodes().getMinNodeVersion();
        } catch (Exception exception) {
            log.error("Cannot get cluster nodes", exception);
        }
        log.debug("Return cluster min version {}", minVersion);
        return minVersion;
    }
}
