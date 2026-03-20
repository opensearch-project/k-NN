/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.opensearch.core.common.Strings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.Version;

import static org.opensearch.knn.common.KNNConstants.ENCODER_FAISS_BBQ;

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

        int belowThresholdDimension = 500; // A dimension below the threshold
        int aboveThresholdDimension = 1500; // A dimension above the threshold

        // x32 with dimension <= 1000 should have an oversample factor of 5.0f
        RescoreContext rescoreContext = CompressionLevel.x32.getDefaultRescoreContext(mode, belowThresholdDimension);
        assertNotNull(rescoreContext);
        assertEquals(5.0f, rescoreContext.getOversampleFactor(), 0.0f);
        assertTrue(rescoreContext.isRescoreEnabled());
        assertFalse(rescoreContext.isUserProvided());

        // x32 with dimension > 1000 should have an oversample factor of 3.0f
        rescoreContext = CompressionLevel.x32.getDefaultRescoreContext(mode, aboveThresholdDimension);
        assertNotNull(rescoreContext);
        assertEquals(3.0f, rescoreContext.getOversampleFactor(), 0.0f);
        assertTrue(rescoreContext.isRescoreEnabled());
        assertFalse(rescoreContext.isUserProvided());

        // x16 with dimension <= 1000 should have an oversample factor of 5.0f
        rescoreContext = CompressionLevel.x16.getDefaultRescoreContext(mode, belowThresholdDimension);
        assertNotNull(rescoreContext);
        assertEquals(5.0f, rescoreContext.getOversampleFactor(), 0.0f);
        assertTrue(rescoreContext.isRescoreEnabled());
        assertFalse(rescoreContext.isUserProvided());

        // x16 with dimension > 1000 should have an oversample factor of 3.0f
        rescoreContext = CompressionLevel.x16.getDefaultRescoreContext(mode, aboveThresholdDimension);
        assertNotNull(rescoreContext);
        assertEquals(3.0f, rescoreContext.getOversampleFactor(), 0.0f);
        assertTrue(rescoreContext.isRescoreEnabled());
        assertFalse(rescoreContext.isUserProvided());

        // x8 with dimension <= 1000 should have an oversample factor of 5.0f
        rescoreContext = CompressionLevel.x8.getDefaultRescoreContext(mode, belowThresholdDimension);
        assertNotNull(rescoreContext);
        assertEquals(5.0f, rescoreContext.getOversampleFactor(), 0.0f);
        assertTrue(rescoreContext.isRescoreEnabled());
        assertFalse(rescoreContext.isUserProvided());

        // x8 with dimension > 1000 should have an oversample factor of 2.0f
        rescoreContext = CompressionLevel.x8.getDefaultRescoreContext(mode, aboveThresholdDimension);
        assertNotNull(rescoreContext);
        assertEquals(2.0f, rescoreContext.getOversampleFactor(), 0.0f);
        assertTrue(rescoreContext.isRescoreEnabled());
        assertFalse(rescoreContext.isUserProvided());

        // x4 with dimension <= 1000 should have an oversample factor of 5.0f
        rescoreContext = CompressionLevel.x4.getDefaultRescoreContext(mode, belowThresholdDimension);
        assertNotNull(rescoreContext);
        assertEquals(1.0f, rescoreContext.getOversampleFactor(), 0.0f);
        assertTrue(rescoreContext.isRescoreEnabled());
        assertFalse(rescoreContext.isUserProvided());
        // x4 with dimension > 1000 should have an oversample factor of 1.0f
        rescoreContext = CompressionLevel.x4.getDefaultRescoreContext(mode, aboveThresholdDimension);
        assertNotNull(rescoreContext);
        assertEquals(1.0f, rescoreContext.getOversampleFactor(), 0.0f);
        assertTrue(rescoreContext.isRescoreEnabled());
        assertFalse(rescoreContext.isUserProvided());
        // Other compression levels should behave similarly with respect to dimension
        rescoreContext = CompressionLevel.x2.getDefaultRescoreContext(mode, belowThresholdDimension);
        assertNull(rescoreContext);
        // x2 with dimension > 1000 should return null
        rescoreContext = CompressionLevel.x2.getDefaultRescoreContext(mode, aboveThresholdDimension);
        assertNull(rescoreContext);
        rescoreContext = CompressionLevel.x1.getDefaultRescoreContext(mode, belowThresholdDimension);
        assertNull(rescoreContext);
        // x1 with dimension > 1000 should return null
        rescoreContext = CompressionLevel.x1.getDefaultRescoreContext(mode, aboveThresholdDimension);
        assertNull(rescoreContext);
        // NOT_CONFIGURED with dimension <= 1000 should return a RescoreContext with an oversample factor of 5.0f
        rescoreContext = CompressionLevel.NOT_CONFIGURED.getDefaultRescoreContext(mode, belowThresholdDimension);
        assertNull(rescoreContext);
    }

    public void testGetDefaultRescoreContext_whenFlatMethod_thenReturnFlatOversampleFactor() {
        // flat method uses x32 compression — mode is NOT_CONFIGURED since flat doesn't support mode
        RescoreContext rescoreContext = CompressionLevel.x32.getDefaultRescoreContext(
            Mode.NOT_CONFIGURED,
            500,
            org.opensearch.Version.CURRENT,
            true
        );
        assertNotNull(rescoreContext);
        assertEquals(2.0f, rescoreContext.getOversampleFactor(), 0.0f);
        assertFalse(rescoreContext.isUserProvided());

        // isFlatMethod=false on x32 with NOT_CONFIGURED mode should return null (no mode for rescore)
        rescoreContext = CompressionLevel.x32.getDefaultRescoreContext(Mode.NOT_CONFIGURED, 500, Version.CURRENT, false);
        assertNull(rescoreContext);
    }

    public void testGetDefaultRescoreContext_whenBBQEncoder_thenReturnFixedOversampleFactor() {
        // BBQ encoder should return fixed oversample factor regardless of compression level, mode, or dimension
        for (CompressionLevel level : CompressionLevel.values()) {
            RescoreContext rescoreContext = level.getDefaultRescoreContext(
                Mode.NOT_CONFIGURED,
                500,
                Version.CURRENT,
                false,
                ENCODER_FAISS_BBQ
            );
            assertNotNull("BBQ rescore context should not be null for " + level, rescoreContext);
            assertEquals(RescoreContext.FAISS_SCALAR_QUANTIZED_INDEX_OVERSAMPLE_FACTOR, rescoreContext.getOversampleFactor(), 0.0f);
            assertFalse(rescoreContext.isUserProvided());
            assertFalse(rescoreContext.isAllowOverrideOversampleFactor());
        }

        // BBQ should also work with ON_DISK mode and high dimension
        RescoreContext rescoreContext = CompressionLevel.x8.getDefaultRescoreContext(
            Mode.ON_DISK,
            1500,
            Version.CURRENT,
            false,
            ENCODER_FAISS_BBQ
        );
        assertNotNull(rescoreContext);
        assertEquals(RescoreContext.FAISS_SCALAR_QUANTIZED_INDEX_OVERSAMPLE_FACTOR, rescoreContext.getOversampleFactor(), 0.0f);
        assertFalse(rescoreContext.isAllowOverrideOversampleFactor());
    }

    public void testGetDefaultRescoreContext_whenNonBBQEncoder_thenFallsBackToNormalLogic() {
        // Non-BBQ encoder should fall through to normal compression level logic
        RescoreContext rescoreContext = CompressionLevel.x8.getDefaultRescoreContext(Mode.ON_DISK, 500, Version.CURRENT, false, "sq");
        assertNotNull(rescoreContext);
        // x8 with dimension <= 1000 should use 5.0f oversample (normal logic)
        assertEquals(RescoreContext.OVERSAMPLE_FACTOR_BELOW_DIMENSION_THRESHOLD, rescoreContext.getOversampleFactor(), 0.0f);

        // null encoder should also fall through
        rescoreContext = CompressionLevel.x8.getDefaultRescoreContext(Mode.ON_DISK, 500, Version.CURRENT, false, null);
        assertNotNull(rescoreContext);
        assertEquals(RescoreContext.OVERSAMPLE_FACTOR_BELOW_DIMENSION_THRESHOLD, rescoreContext.getOversampleFactor(), 0.0f);
    }
}
