/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.nmslib;

import org.opensearch.knn.KNNTestCase;

import java.util.List;

public class NMSLibTests extends KNNTestCase {

    public void testMmapFileExtensions() {
        final List<String> mmapExtensions = Nmslib.INSTANCE.mmapFileExtensions();
        assertNotNull(mmapExtensions);
        final List<String> expectedSettings = List.of("vex", "vec");
        assertTrue(expectedSettings.containsAll(mmapExtensions));
        assertTrue(mmapExtensions.containsAll(expectedSettings));
    }
}
