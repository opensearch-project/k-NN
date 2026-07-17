/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

public class KNNMethodConfigContextTests extends KNNTestCase {

    public void testDeriveMode_whenNotConfigured_thenNotConfigured() {
        assertEquals(Mode.NOT_CONFIGURED, KNNMethodConfigContext.deriveMode(CompressionLevel.NOT_CONFIGURED));
    }

    public void testDeriveMode_whenX1_thenInMemory() {
        assertEquals(Mode.IN_MEMORY, KNNMethodConfigContext.deriveMode(CompressionLevel.x1));
    }

    public void testDeriveMode_whenX2_thenInMemory() {
        assertEquals(Mode.IN_MEMORY, KNNMethodConfigContext.deriveMode(CompressionLevel.x2));
    }

    public void testDeriveMode_whenX4_thenOnDisk() {
        assertEquals(Mode.ON_DISK, KNNMethodConfigContext.deriveMode(CompressionLevel.x4));
    }

    public void testDeriveMode_whenX16_thenOnDisk() {
        assertEquals(Mode.ON_DISK, KNNMethodConfigContext.deriveMode(CompressionLevel.x16));
    }

    public void testDeriveMode_whenX32_thenOnDisk() {
        assertEquals(Mode.ON_DISK, KNNMethodConfigContext.deriveMode(CompressionLevel.x32));
    }

    public void testUserConfiguredCompressionLevel_whenNotResolved_thenMatchesCompressionLevel() {
        KNNMethodConfigContext context = KNNMethodConfigContext.builder().compressionLevel(CompressionLevel.x32).build();
        assertEquals(CompressionLevel.x32, context.getUserConfiguredCompressionLevel());
    }

    public void testUserConfiguredCompressionLevel_whenResolutionOverwrites_thenPreservesOriginal() {
        // Simulates encoder-derived compression: user did not configure compression, but method
        // resolution derived x32 from the encoder (e.g. binary encoder with bits=1)
        KNNMethodConfigContext context = KNNMethodConfigContext.builder().build();
        context.setCompressionLevel(CompressionLevel.x32);
        assertEquals(CompressionLevel.x32, context.getCompressionLevel());
        assertEquals(CompressionLevel.NOT_CONFIGURED, context.getUserConfiguredCompressionLevel());
    }

    public void testUserConfiguredCompressionLevel_whenUserConfiguredAndResolved_thenPreservesUserValue() {
        // User configured x32; resolution re-sets the same value after validating the encoder
        KNNMethodConfigContext context = KNNMethodConfigContext.builder().compressionLevel(CompressionLevel.x32).build();
        context.setCompressionLevel(CompressionLevel.x32);
        assertEquals(CompressionLevel.x32, context.getCompressionLevel());
        assertEquals(CompressionLevel.x32, context.getUserConfiguredCompressionLevel());
    }
}
