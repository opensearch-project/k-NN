/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.Version;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.engine.faiss.Faiss;
import org.opensearch.knn.index.engine.lucene.Lucene;
import org.opensearch.knn.index.engine.nmslib.Nmslib;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class KNNEngineTests extends KNNTestCase {
    /**
     * Check that version from engine and library match
     */
    public void testDelegateLibraryFunctions() {
        assertEquals(Nmslib.INSTANCE.getVersion(), KNNEngine.NMSLIB.getVersion());
        assertEquals(Faiss.INSTANCE.getVersion(), KNNEngine.FAISS.getVersion());
        assertEquals(Lucene.INSTANCE.getVersion(), KNNEngine.LUCENE.getVersion());

        // Validate that deprecated engines have correct deprecation versions
        assertTrue(KNNEngine.NMSLIB.getRestrictedFromVersion() != null);
        assertFalse(KNNEngine.FAISS.isRestricted(Version.V_3_0_0)); // FAISS should not be deprecated
    }

    /**
     * Test that deprecated engines are correctly flagged
     */
    public void testIsRestricted() {
        Version deprecatedVersion = KNNEngine.NMSLIB.getRestrictedFromVersion();
        assertNotNull(deprecatedVersion);
        assertTrue(KNNEngine.NMSLIB.isRestricted(Version.V_3_0_0)); // Should return true for later versions

        assertFalse(KNNEngine.FAISS.isRestricted(Version.V_2_19_0)); // FAISS should not be deprecated
        assertFalse(KNNEngine.LUCENE.isRestricted(Version.V_2_19_0)); // LUCENE should not be deprecated
    }

    public void testGetDefaultEngine_thenReturnFAISS() {
        assertEquals(KNNEngine.FAISS, KNNEngine.DEFAULT);
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

    public void testGetEngineFromPath() {
        String hnswPath1 = "test" + Nmslib.EXTENSION;
        assertEquals(KNNEngine.NMSLIB, KNNEngine.getEngineNameFromPath(hnswPath1));
        String hnswPath2 = "test" + Nmslib.EXTENSION + KNNConstants.COMPOUND_EXTENSION;
        assertEquals(KNNEngine.NMSLIB, KNNEngine.getEngineNameFromPath(hnswPath2));

        String faissPath1 = "test" + KNNConstants.FAISS_EXTENSION;
        assertEquals(KNNEngine.FAISS, KNNEngine.getEngineNameFromPath(faissPath1));
        String faissPath2 = "test" + KNNConstants.FAISS_EXTENSION + KNNConstants.COMPOUND_EXTENSION;
        assertEquals(KNNEngine.FAISS, KNNEngine.getEngineNameFromPath(faissPath2));

        String invalidPath = "test.invalid";
        expectThrows(IllegalArgumentException.class, () -> KNNEngine.getEngineNameFromPath(invalidPath));
    }

    public void testMmapFileExtensions() {
        final List<String> mmapExtensions = Arrays.stream(KNNEngine.values())
            .flatMap(engine -> engine.mmapFileExtensions().stream())
            .collect(Collectors.toList());
        assertNotNull(mmapExtensions);
        final List<String> expectedSettings = List.of("vex", "vec");
        assertTrue(expectedSettings.containsAll(mmapExtensions));
        assertTrue(mmapExtensions.containsAll(expectedSettings));
    }
}
