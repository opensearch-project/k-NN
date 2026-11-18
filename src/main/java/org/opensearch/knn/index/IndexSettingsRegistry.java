/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.opensearch.core.index.Index;
import org.opensearch.index.IndexService;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.shard.IndexEventListener;
import org.opensearch.indices.cluster.IndicesClusterStateService.AllocatedIndices.IndexRemovalReason;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** The live {@link IndexSettings} of each index on this node by UUID, for a codec that has only the shard's path. */
public final class IndexSettingsRegistry implements IndexEventListener {

    private static final IndexSettingsRegistry INSTANCE = new IndexSettingsRegistry();

    private final Map<String, IndexSettings> indexSettingsByUuid = new ConcurrentHashMap<>();

    private IndexSettingsRegistry() {}

    /** {@return the registry of this JVM, shared by its node and its codecs} */
    public static IndexSettingsRegistry getInstance() {
        return INSTANCE;
    }

    @Override
    public void afterIndexCreated(final IndexService indexService) {
        final IndexSettings indexSettings = indexService.getIndexSettings();
        indexSettingsByUuid.put(indexSettings.getUUID(), indexSettings);
    }

    @Override
    public void afterIndexRemoved(final Index index, final IndexSettings indexSettings, final IndexRemovalReason reason) {
        // By identity: a close and open keeps the UUID, and the old index's removal must not drop the new one's entry.
        indexSettingsByUuid.remove(index.getUUID(), indexSettings);
    }

    /**
     * The live settings of the index with this UUID.
     *
     * @param indexUuid The UUID of the index.
     * @return Its live settings, or null when no index on this node has that UUID.
     */
    public IndexSettings get(final String indexUuid) {
        return indexSettingsByUuid.get(indexUuid);
    }
}
