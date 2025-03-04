/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.apache.lucene.index.FieldInfo;
import org.mockito.Mockito;
import org.opensearch.Version;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.engine.faiss.Faiss;
import org.opensearch.knn.index.engine.lucene.Lucene;
import org.opensearch.knn.index.engine.nmslib.Nmslib;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.COMPOUND_EXTENSION;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_EXTENSION;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.NMSLIB_NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.SpaceType.L2;
import static org.opensearch.knn.index.VectorDataType.FLOAT;
import static org.opensearch.knn.index.remote.RemoteIndexHTTPClientTests.MOCK_INDEX_DESCRIPTION;

public class KNNEngineTests extends KNNTestCase {

    public static final String KNN_FIELD = "knn_field";
    public static final String PER_FIELD_KNN_VECTORS_FORMAT_SUFFIX = "PerFieldKnnVectorsFormat.suffix";
    public static final String PER_FIELD_KNN_VECTORS_FORMAT_FORMAT = "PerFieldKnnVectorsFormat.format";

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
        assertEquals(NMSLIB_NAME, KNNEngine.NMSLIB.getName());
    }

    /**
     * Test engine getter
     */
    public void testGetEngine() {
        assertEquals(KNNEngine.NMSLIB, KNNEngine.getEngine(NMSLIB_NAME));
        expectThrows(IllegalArgumentException.class, () -> KNNEngine.getEngine("invalid"));
    }

    public void testGetEngineFromPath() {
        String hnswPath1 = "test" + Nmslib.EXTENSION;
        assertEquals(KNNEngine.NMSLIB, KNNEngine.getEngineNameFromPath(hnswPath1));
        String hnswPath2 = "test" + Nmslib.EXTENSION + COMPOUND_EXTENSION;
        assertEquals(KNNEngine.NMSLIB, KNNEngine.getEngineNameFromPath(hnswPath2));

        String faissPath1 = "test" + FAISS_EXTENSION;
        assertEquals(KNNEngine.FAISS, KNNEngine.getEngineNameFromPath(faissPath1));
        String faissPath2 = "test" + FAISS_EXTENSION + COMPOUND_EXTENSION;
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

    /**
     * The remote build service currently only supports HNSWFlat.
     */
    public void testSupportsRemoteIndexBuild() throws IOException {
        KNNEngine Faiss = KNNEngine.FAISS;
        KNNEngine Lucene = KNNEngine.LUCENE;
        KNNEngine Nmslib = KNNEngine.NMSLIB;

        FieldInfo faissHNSWFlat = createMockFieldInfo(createHnswFlatParameters());
        FieldInfo faissHNSWSQ = createMockFieldInfo(createHnswSQParameters());

        assertTrue(Faiss.supportsRemoteIndexBuild(faissHNSWFlat.attributes()));
        assertFalse(Faiss.supportsRemoteIndexBuild(faissHNSWSQ.attributes()));
        assertFalse(Lucene.supportsRemoteIndexBuild(faissHNSWFlat.attributes()));
        assertFalse(Nmslib.supportsRemoteIndexBuild(faissHNSWFlat.attributes()));
    }

    private FieldInfo createMockFieldInfo(Map<String, String> attributes) {
        FieldInfo fieldInfo = Mockito.mock(FieldInfo.class);
        when(fieldInfo.attributes()).thenReturn(attributes);
        return fieldInfo;
    }

    private Map<String, String> createHnswFlatParameters() {
        Map<String, String> attributes = new HashMap<>();
        attributes.put(KNN_FIELD, "true");
        attributes.put(PER_FIELD_KNN_VECTORS_FORMAT_SUFFIX, "0");
        attributes.put(SPACE_TYPE, L2.getValue());
        attributes.put(KNN_ENGINE, FAISS_NAME);
        attributes.put(VECTOR_DATA_TYPE_FIELD, FLOAT.getValue());
        attributes.put(PER_FIELD_KNN_VECTORS_FORMAT_FORMAT, "NativeEngines990KnnVectorsFormat");
        attributes.put(DIMENSION, "2");
        attributes.put(
            PARAMETERS,
            "{\""
                + INDEX_DESCRIPTION_PARAMETER
                + "\":\""
                + MOCK_INDEX_DESCRIPTION
                + "\",\""
                + METHOD_PARAMETER_SPACE_TYPE
                + "\":\""
                + L2.getValue()
                + "\",\""
                + NAME
                + "\":\""
                + METHOD_HNSW
                + "\",\""
                + VECTOR_DATA_TYPE_FIELD
                + "\":\""
                + FLOAT.getValue()
                + "\","
                + "\""
                + PARAMETERS
                + "\":{\""
                + METHOD_PARAMETER_EF_SEARCH
                + "\":24,\""
                + METHOD_PARAMETER_EF_CONSTRUCTION
                + "\":28,\""
                + METHOD_ENCODER_PARAMETER
                + "\":{\""
                + NAME
                + "\":\""
                + ENCODER_FLAT
                + "\",\""
                + PARAMETERS
                + "\":{}}}}"
        );
        return attributes;
    }

    private Map<String, String> createHnswSQParameters() {
        Map<String, String> attributes = new HashMap<>();
        attributes.put(KNN_FIELD, "true");
        attributes.put(PER_FIELD_KNN_VECTORS_FORMAT_SUFFIX, "0");
        attributes.put(SPACE_TYPE, L2.getValue());
        attributes.put(KNN_ENGINE, FAISS_NAME);
        attributes.put(VECTOR_DATA_TYPE_FIELD, FLOAT.getValue());
        attributes.put(PER_FIELD_KNN_VECTORS_FORMAT_FORMAT, "NativeEngines990KnnVectorsFormat");
        attributes.put(DIMENSION, "2");
        attributes.put(
            PARAMETERS,
            "{\""
                + INDEX_DESCRIPTION_PARAMETER
                + "\":\""
                + MOCK_INDEX_DESCRIPTION
                + "\",\""
                + METHOD_PARAMETER_SPACE_TYPE
                + "\":\""
                + L2.getValue()
                + "\",\""
                + NAME
                + "\":\""
                + METHOD_HNSW
                + "\",\""
                + VECTOR_DATA_TYPE_FIELD
                + "\":\""
                + FLOAT.getValue()
                + "\","
                + "\""
                + PARAMETERS
                + "\":{\""
                + METHOD_PARAMETER_EF_SEARCH
                + "\":24,\""
                + METHOD_PARAMETER_EF_CONSTRUCTION
                + "\":28,\""
                + METHOD_ENCODER_PARAMETER
                + "\":{\""
                + NAME
                + "\":\""
                + ENCODER_SQ
                + "\",\""
                + PARAMETERS
                + "\":{}}}}"
        );
        return attributes;
    }
}
