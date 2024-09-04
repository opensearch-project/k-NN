/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.opensearch.core.common.Strings;
import org.opensearch.knn.KNNTestCase;

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
}
