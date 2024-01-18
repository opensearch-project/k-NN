/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.opensearch.Version;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Setting;

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
     *
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

    /**
     * Get setting value for the cluster. Return default if not set.
     *
     * @param <T> Setting type
     * @return T     setting value or default
     */
    public <T> T getClusterSetting(Setting<T> setting) {
        return clusterService.getClusterSettings().get(setting);
    }

    /**
     * Get index metadata for a particular index
     *
     * @param indexName Name of the index
     * @return IndexMetadata for the given index
     */
    public IndexMetadata getIndexMetadata(String indexName) {
        return clusterService.state().getMetadata().index(indexName);
    }
}
