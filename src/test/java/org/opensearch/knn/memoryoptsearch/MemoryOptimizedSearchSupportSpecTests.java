/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.mockito.MockedStatic;
import org.opensearch.Version;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.exception.MemoryOptimizedSearchOldIndicesNotSupportedException;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.engine.MemoryOptimizedSearchSupportSpec;
import org.opensearch.knn.index.engine.ResolvedIndexSpec;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.when;

public class MemoryOptimizedSearchSupportSpecTests extends KNNTestCase {

    public void testIsSupportedFieldTypeDuringSearch() {

        // When alwaysUseMemoryOptimizedSearch is true, isSupportedFieldType should return true regardless of other settings
        final KNNVectorFieldType alwaysUseFieldType = mock(KNNVectorFieldType.class);
        when(alwaysUseFieldType.isAlwaysUseMemoryOptimizedSearch()).thenReturn(true);
        assertTrue(MemoryOptimizedSearchSupportSpec.isSupportedFieldType(alwaysUseFieldType, "IndexName"));

        // @formatter:off
        /*
        |----------------------|-------------|---------------||-----------|
        | field type supported | mem_opt_src | on_disk && 1x || supported |
        |----------------------|-------------|---------------||-----------|
        |         true         |     true    |      true     ||    true   |
        |         true         |     true    |      false    ||    true   |
        |         true         |     false   |      true     ||    true   |
        |         true         |     false   |      false    ||    false  |
        |         false        |     true    |      true     ||    false  |
        |         false        |     true    |      false    ||    false  |
        |         false        |     false   |      true     ||    false  |
        |         false        |     false   |      false    ||    false  |
        |----------------------|-------------|---------------||-----------|
        */
        // @formatter:on

        doTestIsSupportedFieldTypeDuringSearch(true, true, true, true, Version.CURRENT);
        doTestIsSupportedFieldTypeDuringSearch(true, true, false, true, Version.CURRENT);
        doTestIsSupportedFieldTypeDuringSearch(true, false, true, true, Version.CURRENT);
        doTestIsSupportedFieldTypeDuringSearch(true, false, false, false, Version.CURRENT);
        doTestIsSupportedFieldTypeDuringSearch(false, true, true, false, Version.CURRENT);
        doTestIsSupportedFieldTypeDuringSearch(false, true, false, false, Version.CURRENT);
        doTestIsSupportedFieldTypeDuringSearch(false, false, true, false, Version.CURRENT);
        doTestIsSupportedFieldTypeDuringSearch(false, false, false, false, Version.CURRENT);
    }

    public void testShouldThrowExceptionRegardless() {
        // @formatter:off
        /*
        |----------------------|-------------|---------------|----------||-----------|
        | field type supported | mem_opt_src | on_disk && 1x |  version || supported |
        |----------------------|-------------|---------------|----------||-----------|
        |         true         |     true    |      true     |    old   ||    false  |
        |         true         |     true    |      false    |    old   ||    false  |
        |         true         |     false   |      true     |    old   ||    false  |
        |         true         |     false   |      false    |    old   ||    false  |
        |         false        |     true    |      true     |    old   ||    false  |
        |         false        |     true    |      false    |    old   ||    false  |
        |         false        |     false   |      true     |    old   ||    false  |
        |         false        |     false   |      false    |    old   ||    false  |
        |----------------------|-------------|---------------|----------||-----------|
        */
        // @formatter:on

        assertThrows(
            MemoryOptimizedSearchOldIndicesNotSupportedException.class,
            () -> doTestIsSupportedFieldTypeDuringSearch(true, true, true, true, Version.V_2_16_0)
        );
        assertThrows(
            MemoryOptimizedSearchOldIndicesNotSupportedException.class,
            () -> doTestIsSupportedFieldTypeDuringSearch(true, true, false, true, Version.V_2_16_0)
        );

        // It's ok! MemOptSrch is turned off.
        doTestIsSupportedFieldTypeDuringSearch(false, true, false, false, Version.V_2_16_0);
        doTestIsSupportedFieldTypeDuringSearch(false, true, true, false, Version.V_2_16_0);
        doTestIsSupportedFieldTypeDuringSearch(false, false, true, false, Version.V_2_16_0);
        doTestIsSupportedFieldTypeDuringSearch(false, false, false, false, Version.V_2_16_0);
        doTestIsSupportedFieldTypeDuringSearch(false, false, true, false, Version.V_2_16_0);
        doTestIsSupportedFieldTypeDuringSearch(false, false, false, false, Version.V_2_16_0);
    }

    public void doTestIsSupportedFieldTypeDuringSearch(
        final boolean fieldTypeSupported,
        final boolean memoryOptSrchSupported,
        final boolean onDiskWith1x,
        final boolean expected,
        final Version version
    ) {
        try (MockedStatic<KNNSettings> knnSettingsMockedStatic = mockStatic(KNNSettings.class)) {
            knnSettingsMockedStatic.when(() -> KNNSettings.isMemoryOptimizedKnnSearchModeEnabled(any())).thenReturn(memoryOptSrchSupported);

            final KNNVectorFieldType fieldType = mock(KNNVectorFieldType.class);
            when(fieldType.getIndexCreatedVersion()).thenReturn(version);
            when(fieldType.isMemoryOptimizedSearchAvailable()).thenReturn(fieldTypeSupported);

            final ResolvedIndexSpec resolvedSpec = mock(ResolvedIndexSpec.class);
            when(resolvedSpec.requiresMemoryOptimizedSearchForOnDisk()).thenReturn(onDiskWith1x);
            when(fieldType.getResolvedSpec()).thenReturn(resolvedSpec);

            assertEquals(expected, MemoryOptimizedSearchSupportSpec.isSupportedFieldType(fieldType, "IndexName"));
        }
    }
}
