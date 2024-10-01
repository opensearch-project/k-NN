/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.opensearch.core.common.Strings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.query.rescore.RescoreContext;

public class CompressionLevelTests extends KNNTestCase {

    public void testFromName() {
        assertEquals(CompressionLevel.NOT_CONFIGURED, CompressionLevel.fromName(null));
        assertEquals(CompressionLevel.NOT_CONFIGURED, CompressionLevel.fromName(""));
        assertEquals(CompressionLevel.x1, CompressionLevel.fromName("1x"));
        assertEquals(CompressionLevel.x32, CompressionLevel.fromName("32x"));
        expectThrows(IllegalArgumentException.class, () -> CompressionLevel.fromName("x1"));
    }

    public void testGetName() {
        assertTrue(Strings.isEmpty(CompressionLevel.NOT_CONFIGURED.getName()));
        assertEquals("4x", CompressionLevel.x4.getName());
        assertEquals("16x", CompressionLevel.x16.getName());
    }

    public void testNumBitsForFloat32() {
        assertEquals(1, CompressionLevel.x32.numBitsForFloat32());
        assertEquals(2, CompressionLevel.x16.numBitsForFloat32());
        assertEquals(4, CompressionLevel.x8.numBitsForFloat32());
        assertEquals(8, CompressionLevel.x4.numBitsForFloat32());
        assertEquals(16, CompressionLevel.x2.numBitsForFloat32());
        assertEquals(32, CompressionLevel.x1.numBitsForFloat32());
        assertEquals(32, CompressionLevel.NOT_CONFIGURED.numBitsForFloat32());
    }

    public void testIsConfigured() {
        assertFalse(CompressionLevel.isConfigured(CompressionLevel.NOT_CONFIGURED));
        assertFalse(CompressionLevel.isConfigured(null));
        assertTrue(CompressionLevel.isConfigured(CompressionLevel.x1));
    }

    public void testGetDefaultRescoreContext() {
        // Test rescore context for ON_DISK mode
        Mode mode = Mode.ON_DISK;

        // Test various dimensions based on the updated oversampling logic
        int belowThresholdDimension = 500;  // A dimension below 768
        int between768and1000Dimension = 800; // A dimension between 768 and 1000
        int above1000Dimension = 1500; // A dimension above 1000

        // Compression level x32 with dimension < 768 should have an oversample factor of 3.0f
        RescoreContext rescoreContext = CompressionLevel.x32.getDefaultRescoreContext(mode, belowThresholdDimension);
        assertNotNull(rescoreContext);
        assertEquals(3.0f, rescoreContext.getOversampleFactor(), 0.0f);

        // Compression level x32 with dimension between 768 and 1000 should have an oversample factor of 2.0f
        rescoreContext = CompressionLevel.x32.getDefaultRescoreContext(mode, between768and1000Dimension);
        assertNotNull(rescoreContext);
        assertEquals(2.0f, rescoreContext.getOversampleFactor(), 0.0f);

        // Compression level x32 with dimension > 1000 should have no oversampling (1.0f)
        rescoreContext = CompressionLevel.x32.getDefaultRescoreContext(mode, above1000Dimension);
        assertNotNull(rescoreContext);
        assertEquals(1.0f, rescoreContext.getOversampleFactor(), 0.0f);

        // Compression level x16 with dimension < 768 should have an oversample factor of 3.0f
        rescoreContext = CompressionLevel.x16.getDefaultRescoreContext(mode, belowThresholdDimension);
        assertNotNull(rescoreContext);
        assertEquals(3.0f, rescoreContext.getOversampleFactor(), 0.0f);

        // Compression level x16 with dimension between 768 and 1000 should have an oversample factor of 2.0f
        rescoreContext = CompressionLevel.x16.getDefaultRescoreContext(mode, between768and1000Dimension);
        assertNotNull(rescoreContext);
        assertEquals(2.0f, rescoreContext.getOversampleFactor(), 0.0f);

        // Compression level x16 with dimension > 1000 should have no oversampling (1.0f)
        rescoreContext = CompressionLevel.x16.getDefaultRescoreContext(mode, above1000Dimension);
        assertNotNull(rescoreContext);
        assertEquals(1.0f, rescoreContext.getOversampleFactor(), 0.0f);

        // Compression level x8 with dimension < 768 should have an oversample factor of 3.0f
        rescoreContext = CompressionLevel.x8.getDefaultRescoreContext(mode, belowThresholdDimension);
        assertNotNull(rescoreContext);
        assertEquals(3.0f, rescoreContext.getOversampleFactor(), 0.0f);

        // Compression level x8 with dimension between 768 and 1000 should have an oversample factor of 2.0f
        rescoreContext = CompressionLevel.x8.getDefaultRescoreContext(mode, between768and1000Dimension);
        assertNotNull(rescoreContext);
        assertEquals(2.0f, rescoreContext.getOversampleFactor(), 0.0f);

        // Compression level x8 with dimension > 1000 should have no oversampling (1.0f)
        rescoreContext = CompressionLevel.x8.getDefaultRescoreContext(mode, above1000Dimension);
        assertNotNull(rescoreContext);
        assertEquals(1.0f, rescoreContext.getOversampleFactor(), 0.0f);

        // Compression level x4 with dimension < 768 should return null (no RescoreContext)
        rescoreContext = CompressionLevel.x4.getDefaultRescoreContext(mode, belowThresholdDimension);
        assertNull(rescoreContext);

        // Compression level x4 with dimension > 1000 should return null (no RescoreContext)
        rescoreContext = CompressionLevel.x4.getDefaultRescoreContext(mode, above1000Dimension);
        assertNull(rescoreContext);

        // Compression level x2 with dimension < 768 should return null
        rescoreContext = CompressionLevel.x2.getDefaultRescoreContext(mode, belowThresholdDimension);
        assertNull(rescoreContext);

        // Compression level x2 with dimension > 1000 should return null
        rescoreContext = CompressionLevel.x2.getDefaultRescoreContext(mode, above1000Dimension);
        assertNull(rescoreContext);

        // Compression level x1 with dimension < 768 should return null
        rescoreContext = CompressionLevel.x1.getDefaultRescoreContext(mode, belowThresholdDimension);
        assertNull(rescoreContext);

        // Compression level x1 with dimension > 1000 should return null
        rescoreContext = CompressionLevel.x1.getDefaultRescoreContext(mode, above1000Dimension);
        assertNull(rescoreContext);

        // NOT_CONFIGURED mode should return null for any dimension
        rescoreContext = CompressionLevel.NOT_CONFIGURED.getDefaultRescoreContext(mode, belowThresholdDimension);
        assertNull(rescoreContext);
    }

}
