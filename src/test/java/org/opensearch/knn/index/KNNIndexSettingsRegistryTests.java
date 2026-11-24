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

import java.util.function.BooleanSupplier;

public class KNNIndexSettingsRegistryTests extends KNNTestCase {

    private static final String INDEX_UUID = "V4qM8m9ZTAyRm0Nn0lOAbw";

    private final KNNIndexSettingsRegistry registry = KNNIndexSettingsRegistry.getInstance();

    public void testTheSupplierReadsEachUpdate() {
        final IndexSettings indexSettings = created(INDEX_UUID, false);
        try {
            final BooleanSupplier directIOEnabled = registry.directIOEnabled(INDEX_UUID).orElseThrow();
            assertFalse(directIOEnabled.getAsBoolean());
            indexSettings.updateIndexMetadata(indexMetadata(INDEX_UUID, true));
            assertTrue(directIOEnabled.getAsBoolean());
            indexSettings.updateIndexMetadata(indexMetadata(INDEX_UUID, false));
            assertFalse(directIOEnabled.getAsBoolean());
        } finally {
            removed(indexSettings);
        }
    }

    public void testRemovalKeepsANewerRegistrationUnderTheSameUuid() {
        final IndexSettings closed = created(INDEX_UUID, true);
        final IndexSettings reopened = created(INDEX_UUID, false);
        try {
            removed(closed);
            assertFalse(
                "the reopened index's value, not the closed one's",
                registry.directIOEnabled(INDEX_UUID).orElseThrow().getAsBoolean()
            );
            removed(reopened);
            assertTrue(registry.directIOEnabled(INDEX_UUID).isEmpty());
        } finally {
            removed(closed);
            removed(reopened);
        }
    }

    public void testAnUnknownUuidFindsNothing() {
        final IndexSettings indexSettings = created(INDEX_UUID, true);
        try {
            assertTrue(registry.directIOEnabled(INDEX_UUID).orElseThrow().getAsBoolean());
            assertTrue(registry.directIOEnabled("ZZZZZZZZZZZZZZZZZZZZZZ").isEmpty());
        } finally {
            removed(indexSettings);
        }
    }

    public void testAnIndexWithoutTheSettingReadsItsDefault() {
        final IndexSettings indexSettings = created(INDEX_UUID, null);
        try {
            assertEquals(
                KNNSettings.INDEX_KNN_DIRECT_IO_RESCORE_ENABLED_DEFAULT_VALUE,
                registry.directIOEnabled(INDEX_UUID).orElseThrow().getAsBoolean()
            );
        } finally {
            removed(indexSettings);
        }
    }

    private IndexSettings created(final String uuid, final Boolean enabled) {
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

    // A null value leaves the setting out, as for an index created without it.
    private static IndexMetadata indexMetadata(final String uuid, final Boolean enabled) {
        final Settings.Builder settings = Settings.builder()
            .put(IndexMetadata.SETTING_VERSION_CREATED, Version.CURRENT.id)
            .put(IndexMetadata.SETTING_INDEX_UUID, uuid);
        if (enabled != null) {
            settings.put(KNNSettings.INDEX_KNN_DIRECT_IO_RESCORE_ENABLED, enabled);
        }
        return IndexMetadata.builder("target-index").settings(settings.build()).numberOfShards(1).numberOfReplicas(0).build();
    }
}
