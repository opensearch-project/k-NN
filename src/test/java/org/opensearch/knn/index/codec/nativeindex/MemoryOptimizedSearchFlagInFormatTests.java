/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.SneakyThrows;
import org.opensearch.common.settings.Settings;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;

import java.lang.reflect.Field;
import java.util.Optional;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.KNNSettings.DEFAULT_MEMORY_OPTIMIZED_KNN_SEARCH_MODE;
import static org.opensearch.knn.index.KNNSettings.MEMORY_OPTIMIZED_KNN_SEARCH_MODE;

public class MemoryOptimizedSearchFlagInFormatTests extends KNNTestCase {
    private static final int DONT_CARE = -1;

    @SneakyThrows
    public void testWhenNullSettings() {
        // Create format with null index settings
        final IndexSettings indexSettings = mock(IndexSettings.class);
        when(indexSettings.getSettings()).thenReturn(null);

        // Mock MapperService
        final MapperService mapperService = mock(MapperService.class);
        when(mapperService.getIndexSettings()).thenReturn(indexSettings);

        final NativeEngines990KnnVectorsFormat format = new NativeEngines990KnnVectorsFormat(
            null,  // Don't care
            -1,  // Don't care
            null,  // Don't care
            Optional.of(mapperService)
        );

        doTest(format, false);
    }

    @SneakyThrows
    public void testWhenSettingsDontHaveTheFlag() {
        // Mock settings
        final Settings settings = mock(Settings.class);
        when(settings.getAsBoolean(any(), any())).thenReturn(false);

        // Mock index settings
        final IndexSettings indexSettings = mock(IndexSettings.class);
        when(indexSettings.getSettings()).thenReturn(settings);

        // Create format with null index settings
        final NativeEngines990KnnVectorsFormat format = new NativeEngines990KnnVectorsFormat(
            null,  // Don't care
            -1,  // Don't care
            null,  // Don't care
            Optional.empty()
        );

        doTest(format, false);
    }

    @SneakyThrows
    public void testWhenSettingsHaveTheFlag() {
        // Mock settings
        final Settings settings = mock(Settings.class);
        when(settings.getAsBoolean(MEMORY_OPTIMIZED_KNN_SEARCH_MODE, DEFAULT_MEMORY_OPTIMIZED_KNN_SEARCH_MODE)).thenReturn(true);

        // Mock index settings
        final IndexSettings indexSettings = mock(IndexSettings.class);
        when(indexSettings.getSettings()).thenReturn(settings);

        // Mock MapperService
        final MapperService mapperService = mock(MapperService.class);
        when(mapperService.getIndexSettings()).thenReturn(indexSettings);

        // Create format with null index settings
        final NativeEngines990KnnVectorsFormat format = new NativeEngines990KnnVectorsFormat(
            null,  // Don't care
            -1,  // Don't care
            null,  // Don't care
            Optional.of(mapperService)
        );

        doTest(format, true);
    }

    @SneakyThrows
    private void doTest(final NativeEngines990KnnVectorsFormat format, final boolean expected) {
        // Get field
        final Field field = NativeEngines990KnnVectorsFormat.class.getDeclaredField("memoryOptimizedSearchEnabled");
        field.setAccessible(true); // Bypass private access

        // Test whether it is supported
        final boolean result = (boolean) field.get(format);
        assertEquals(expected, result);
    }
}
