/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.opensearch.core.index.Index;
import org.opensearch.index.IndexService;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.shard.IndexEventListener;
import org.opensearch.indices.cluster.IndicesClusterStateService.AllocatedIndices.IndexRemovalReason;

import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BooleanSupplier;

/** The live Direct I/O setting of each index on this node by UUID, for a codec that has only the shard's path. */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class KNNIndexSettingsRegistry implements IndexEventListener {

    private static final KNNIndexSettingsRegistry INSTANCE = new KNNIndexSettingsRegistry();

    private final Map<String, IndexSettings> indexSettingsByUuid = new ConcurrentHashMap<>();

    /** {@return the registry of this JVM, shared by its node and its codecs} */
    public static KNNIndexSettingsRegistry getInstance() {
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
     * The live value of {@code index.knn.direct_io.rescore.enabled} of the index with this UUID.
     *
     * @param indexUuid The UUID of the index.
     * @return A supplier that reads the index's current value at each call, or empty when no index on this node has
     *         that UUID.
     */
    public Optional<BooleanSupplier> directIOEnabled(final String indexUuid) {
        final IndexSettings indexSettings = indexSettingsByUuid.get(indexUuid);
        if (indexSettings == null) {
            return Optional.empty();
        }
        // getSettings() reads the volatile field a settings update replaces, so the next call sees the update.
        return Optional.of(() -> KNNSettings.INDEX_KNN_DIRECT_IO_RESCORE_ENABLED_SETTING.get(indexSettings.getSettings()));
    }
}
