/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.mockito.Mockito;
import org.opensearch.Version;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.common.settings.Settings;
import org.opensearch.index.IndexService;
import org.opensearch.index.IndexSettings;
import org.opensearch.indices.cluster.IndicesClusterStateService.AllocatedIndices.IndexRemovalReason;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.test.IndexSettingsModule;

public class IndexSettingsRegistryTests extends KNNTestCase {

    private static final String INDEX_UUID = "V4qM8m9ZTAyRm0Nn0lOAbw";

    private final IndexSettingsRegistry registry = IndexSettingsRegistry.getInstance();

    public void testAnUpdateReachesTheHeldIndexSettings() {
        final IndexSettings indexSettings = created(INDEX_UUID, false);
        try {
            final IndexSettings held = registry.get(INDEX_UUID);
            assertSame(indexSettings, held);
            assertFalse(KNNSettings.INDEX_KNN_DIRECT_IO_RESCORE_ENABLED_SETTING.get(held.getSettings()));
            indexSettings.updateIndexMetadata(indexMetadata(INDEX_UUID, true));
            assertTrue(KNNSettings.INDEX_KNN_DIRECT_IO_RESCORE_ENABLED_SETTING.get(held.getSettings()));
        } finally {
            removed(indexSettings);
        }
    }

    public void testRemovalKeepsANewerRegistrationUnderTheSameUuid() {
        final IndexSettings closed = created(INDEX_UUID, true);
        final IndexSettings reopened = created(INDEX_UUID, false);
        try {
            removed(closed);
            assertSame(reopened, registry.get(INDEX_UUID));
            removed(reopened);
            assertNull(registry.get(INDEX_UUID));
        } finally {
            removed(closed);
            removed(reopened);
        }
    }

    public void testAnUnknownUuidFindsNothing() {
        final IndexSettings indexSettings = created(INDEX_UUID, true);
        try {
            assertSame(indexSettings, registry.get(INDEX_UUID));
            assertNull(registry.get("ZZZZZZZZZZZZZZZZZZZZZZ"));
        } finally {
            removed(indexSettings);
        }
    }

    private IndexSettings created(final String uuid, final boolean enabled) {
        final IndexSettings indexSettings = IndexSettingsModule.newIndexSettings(
            indexMetadata(uuid, enabled),
            KNNSettings.INDEX_KNN_DIRECT_IO_RESCORE_ENABLED_SETTING
        );
        final IndexService indexService = Mockito.mock(IndexService.class);
        Mockito.when(indexService.getIndexSettings()).thenReturn(indexSettings);
        registry.afterIndexCreated(indexService);
        return indexSettings;
    }

    private void removed(final IndexSettings indexSettings) {
        registry.afterIndexRemoved(indexSettings.getIndex(), indexSettings, IndexRemovalReason.DELETED);
    }

    private static IndexMetadata indexMetadata(final String uuid, final boolean enabled) {
        final Settings settings = Settings.builder()
            .put(IndexMetadata.SETTING_VERSION_CREATED, Version.CURRENT.id)
            .put(IndexMetadata.SETTING_INDEX_UUID, uuid)
            .put(KNNSettings.INDEX_KNN_DIRECT_IO_RESCORE_ENABLED, enabled)
            .build();
        return IndexMetadata.builder("target-index").settings(settings).numberOfShards(1).numberOfReplicas(0).build();
    }
}
