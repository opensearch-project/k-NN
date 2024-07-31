/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.util;

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
public class KNNClusterUtil {

    private ClusterService clusterService;
    private static KNNClusterUtil instance;

    /**
     * Return instance of the cluster context, must be initialized first for proper usage
     * @return instance of cluster context
     */
    public static synchronized KNNClusterUtil instance() {
        if (instance == null) {
            instance = new KNNClusterUtil();
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
        try {
            return this.clusterService.state().getNodes().getMinNodeVersion();
        } catch (Exception exception) {
            log.error(
                String.format("Failed to get cluster minimum node version, returning current node version %s instead.", Version.CURRENT),
                exception
            );
            return Version.CURRENT;
        }
    }
}
