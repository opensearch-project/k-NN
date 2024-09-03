/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.opensearch.core.common.Strings;
import org.opensearch.knn.KNNTestCase;

public class ModeTests extends KNNTestCase {

    public void testFromName() {
        assertEquals(Mode.NOT_CONFIGURED, Mode.fromName(null));
        assertEquals(Mode.NOT_CONFIGURED, Mode.fromName(""));
        assertEquals(Mode.ON_DISK, Mode.fromName("on_disk"));
        assertEquals(Mode.IN_MEMORY, Mode.fromName("in_memory"));
        expectThrows(IllegalArgumentException.class, () -> Mode.fromName("on_disk2"));
    }

    public void testGetName() {
        assertTrue(Strings.isEmpty(Mode.NOT_CONFIGURED.getName()));
        assertEquals("on_disk", Mode.ON_DISK.getName());
        assertEquals("in_memory", Mode.IN_MEMORY.getName());
    }

    public void testIsConfigured() {
        assertFalse(Mode.isConfigured(Mode.NOT_CONFIGURED));
        assertFalse(Mode.isConfigured(null));
        assertTrue(Mode.isConfigured(Mode.ON_DISK));
    }

}
