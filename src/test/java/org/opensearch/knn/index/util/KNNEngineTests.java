/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.util;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;

public class KNNEngineTests extends KNNTestCase {
    /**
     * Get latest build version from library
     */
    public void testDelegateLibraryFunctions() {
        assertEquals(KNNLibrary.Nmslib.INSTANCE.getLatestLibVersion(), KNNEngine.NMSLIB.getLatestLibVersion());
    }

    /**
     * Test name getter
     */
    public void testGetName() {
        assertEquals(KNNConstants.NMSLIB_NAME, KNNEngine.NMSLIB.getName());
    }

    /**
     * Test engine getter
     */
    public void testGetEngine() {
        assertEquals(KNNEngine.NMSLIB, KNNEngine.getEngine(KNNConstants.NMSLIB_NAME));
        expectThrows(IllegalArgumentException.class, () -> KNNEngine.getEngine("invalid"));
    }
}
